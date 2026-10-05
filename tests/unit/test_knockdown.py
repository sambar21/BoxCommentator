import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core import clock
from src.core.action_buffer.buffer import ActionBuffer, Punch
from src.core.cooldown.manager import CooldownManager
from src.core.events import Event
from src.core.orchestrator import CommentaryOrchestrator
from src.core.queue_consumer.consumer import QueueConsumer
from src.trackers.knockdown import KnockdownTracker

from bench.fights import make_fight, make_suite, replay
from bench.quality import knockdown_recall, aggregate
from bench.stub_llm import StubLLM


def _punch(knockdown=False, attacker=1):
    return Punch(attacker=attacker, punch_type="hook", target="head",
                 outcome="landed", timestamp=clock.now(), damage=50, knockdown=knockdown)


def test_tracker_emits_once_per_knockdown():
    buf = ActionBuffer()
    tracker = KnockdownTracker(buf)
    buf.add_punch(_punch(knockdown=False))
    assert tracker.update() is None

    buf.add_punch(_punch(knockdown=True, attacker=2))
    event = tracker.update()
    assert event.type == "knockdown"
    assert event.priority >= 9.5
    assert event.context["attacker"] == 2 and event.context["victim"] == 1
    assert event.context["knockdown_number"] == 1

    # update() runs on every punch; the same punch must not re-emit
    assert tracker.update() is None


def test_tracker_uses_fighter_names():
    buf = ActionBuffer()
    tracker = KnockdownTracker(buf, names=lambda: {1: "Alvarez", 2: "Garcia"})
    buf.add_punch(_punch(knockdown=True, attacker=1))
    assert "Alvarez drops Garcia" in tracker.update().message


def test_knockdown_bypasses_cooldown():
    mgr = CooldownManager()
    first = Event("knockdown", 10.0, "KD 1", {})
    mgr.activate(first)
    assert mgr.check(Event("knockdown", 10.0, "KD 2", {})) is True


def test_consumer_never_demotes_knockdown():
    consumer = QueueConsumer()
    # Force Track B over its 45% share so ordinary B events get demoted
    consumer.track_a_count, consumer.track_b_count = 3, 7
    assert consumer._apply_balance('B', 9.5) == 'A'      # ordinary urgent event: demoted
    assert consumer._apply_balance('B', 10.0) == 'B'     # knockdown: protected


def test_replay_calls_every_knockdown():
    fight = make_fight(seed=1, n_punches=120, n_knockdowns=2)
    orch = CommentaryOrchestrator(llm_client=StubLLM(), use_rag=False)
    replay(fight, orch)
    score = knockdown_recall(fight, orch.generation_log)
    assert score.total == 2
    assert score.called == 2
    assert score.described == 2
    assert all(r.ttft_ms is not None for r in orch.generation_log if r.text)


def test_recall_detects_dropped_knockdowns():
    fight = make_fight(seed=1, n_punches=120, n_knockdowns=2)
    orch = CommentaryOrchestrator(llm_client=StubLLM(drop_knockdowns=True), use_rag=False)
    replay(fight, orch)
    assert knockdown_recall(fight, orch.generation_log).called == 0


def test_replay_restores_real_clock():
    fight = make_fight(seed=2, n_punches=40, n_knockdowns=1)
    replay(fight, CommentaryOrchestrator(llm_client=StubLLM(), use_rag=False))
    assert abs(clock.now() - __import__("time").time()) < 5


def test_suite_recall_all_knockdowns():
    scores = []
    for fight in make_suite(6):
        orch = CommentaryOrchestrator(llm_client=StubLLM(), use_rag=False)
        replay(fight, orch)
        scores.append(knockdown_recall(fight, orch.generation_log))
    total = aggregate(scores)
    assert total.total > 0
    assert total.called == total.total


def test_fight_is_deterministic():
    a = make_fight(seed=7, n_punches=80, n_knockdowns=2)
    b = make_fight(seed=7, n_punches=80, n_knockdowns=2)
    assert a == b


def test_track_a_prompts_share_a_stable_prefix():
    """Prompts within one fight must share a long prefix so vLLM prefix caching can hit."""
    import os
    fight = make_fight(seed=3, n_punches=150, n_knockdowns=1)

    class Capture(StubLLM):
        prompts = []
        def generate_streaming(self, prompt, **kw):
            Capture.prompts.append(prompt)
            yield from super().generate_streaming(prompt, **kw)

    Capture.prompts = []
    orch = CommentaryOrchestrator(llm_client=Capture(), use_rag=False)
    replay(fight, orch)

    a_prompts = [p for p in Capture.prompts if "analyst" in p]
    assert len(a_prompts) > 5
    prefix = os.path.commonprefix(a_prompts)
    assert "FIGHTER PROFILES" in prefix
    assert "--- LIVE SITUATION ---" in prefix
    assert len(prefix) > 600                       # profiles are inside the shared prefix
    # volatile fields must come after the shared prefix
    assert prefix.index("--- LIVE SITUATION ---") < len(prefix)
    assert "Round" not in prefix.split("--- LIVE SITUATION ---")[0]

"""
Quality checks against ground truth.

knockdown_recall(fight, log): did the commentator call every knockdown?
  - called:    a non-empty Track B generation exists for knockdown #k
  - described: that text actually reads as a knockdown call

invented_stats(output, prompt): numbers the model stated that were never given to it.
  The prompt is the ground truth (fighter profiles, live stats, event text), so
  anything numeric in the output that is not in the prompt was made up or
  computed by the model. Words ("three") are not checked, only digits.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Set

from bench.fights import Fight

KNOCKDOWN_WORDS = re.compile(
    r"knock|down|canvas|floor|drop|dropped|drops|flatten|hit the mat|on the mat|count",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Digits that appear in nearly any commentary without being a statistic:
# "one-two combination", fighter 1 / 2.
_ALWAYS_OK = {"1", "2"}


# ── knockdown recall ──────────────────────────────────────────────────────────

@dataclass
class KnockdownScore:
    total: int
    called: int
    described: int

    @property
    def recall_called(self) -> float:
        return self.called / self.total if self.total else 1.0

    @property
    def recall_described(self) -> float:
        return self.described / self.total if self.total else 1.0


def knockdown_recall(fight: Fight, log: Iterable) -> KnockdownScore:
    by_number = {}
    for rec in log:
        if rec.event_type == "knockdown":
            number = rec.event_context.get("knockdown_number")
            by_number[number] = rec

    called = described = 0
    for k in range(1, fight.n_knockdowns + 1):
        rec = by_number.get(k)
        if rec is None or not rec.text.strip():
            continue
        called += 1
        if KNOCKDOWN_WORDS.search(rec.text):
            described += 1
    return KnockdownScore(total=fight.n_knockdowns, called=called, described=described)


def aggregate(scores: List[KnockdownScore]) -> KnockdownScore:
    return KnockdownScore(
        total=sum(s.total for s in scores),
        called=sum(s.called for s in scores),
        described=sum(s.described for s in scores),
    )


# ── invented stats ────────────────────────────────────────────────────────────

def _norm(num: str) -> str:
    """'70.50' -> '70.5', '60.0' -> '60'."""
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def extract_numbers(text: str) -> Set[str]:
    return {_norm(m) for m in _NUMBER.findall(text or "")}


def invented_stats(output: str, prompt: str) -> List[str]:
    """
    Numbers in `output` that are not grounded in `prompt`.

    A fraction in [0, 1] is accepted when its percentage form is in the prompt
    (the model said 0.78, the profile said 78%).
    """
    allowed = extract_numbers(prompt) | _ALWAYS_OK
    invented = []
    for num in sorted(extract_numbers(output)):
        if num in allowed:
            continue
        try:
            as_percent = _norm(f"{float(num) * 100:.1f}")
        except ValueError:
            as_percent = None
        if as_percent in allowed and float(num) <= 1.0:
            continue
        invented.append(num)
    return invented


@dataclass
class StatScore:
    generations: int          # generations with text and a prompt to check against
    with_invented: int        # generations containing at least one ungrounded number
    invented_numbers: int     # total ungrounded numbers

    @property
    def rate(self) -> float:
        """Share of checked generations that contain an invented number."""
        return self.with_invented / self.generations if self.generations else 0.0

    @property
    def per_100(self) -> float:
        return 100.0 * self.rate


def score_stats(log: Iterable) -> StatScore:
    """Needs records produced with keep_prompts=True; records without a prompt are skipped."""
    generations = with_invented = invented_numbers = 0
    for rec in log:
        if not rec.text.strip() or not rec.prompt:
            continue
        generations += 1
        found = invented_stats(rec.text, rec.prompt)
        if found:
            with_invented += 1
            invented_numbers += len(found)
    return StatScore(generations, with_invented, invented_numbers)


def merge_stat_scores(scores: List[StatScore]) -> StatScore:
    return StatScore(
        generations=sum(s.generations for s in scores),
        with_invented=sum(s.with_invented for s in scores),
        invented_numbers=sum(s.invented_numbers for s in scores),
    )

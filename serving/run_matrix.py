"""
Start vLLM variants one at a time and run the benchmark experiments against each.

  python -m serving.run_matrix --plan core --dry-plan        # print what would run
  python -m serving.run_matrix --plan core --fights 10       # run it (needs a GPU + vllm)
  python -m serving.run_matrix --serve-only 3b               # just start a server and wait

Bench tags and the comparison each one feeds (see bench/charts.py):
  baseline      one fight at a time -> quantization (3b vs 3b-awq vs 7b-awq),
                prefix caching on/off (3b vs 3b-nocache), batching point c=1
  batching      4 / 16 concurrent fights (3b)
  caching_full  caching on/off with the whole transcript in the prompt
  capping       long fights: full transcript + cache vs capped transcript, with and without cache

VERIFY on the machine you run this on: vLLM's current flag names (--enable-prefix-caching /
--no-enable-prefix-caching, --quantization awq, --enable-lora) and T4 support.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "serving" / "logs"


@dataclass(frozen=True)
class Variant:
    model: str                   # HF id served (must match the model in backends.yaml)
    backend: str                 # backends.yaml entry the benchmark should call
    flags: Tuple[str, ...] = ()
    gpus: int = 1
    label: Optional[str] = None  # results label (defaults to the variant name)


VARIANTS: Dict[str, Variant] = {
    "3b":         Variant("Qwen/Qwen2.5-3B-Instruct", "vllm-3b", ("--enable-prefix-caching",)),
    "3b-nocache": Variant("Qwen/Qwen2.5-3B-Instruct", "vllm-3b", ("--no-enable-prefix-caching",)),
    "3b-awq":     Variant("Qwen/Qwen2.5-3B-Instruct-AWQ", "vllm-3b-awq",
                          ("--enable-prefix-caching", "--quantization", "awq")),
    "7b-awq":     Variant("Qwen/Qwen2.5-7B-Instruct-AWQ", "vllm-7b-awq",
                          ("--enable-prefix-caching", "--quantization", "awq")),
    # fp16 7B needs ~15 GB for weights alone: only with two GPUs
    "7b-tp2":     Variant("Qwen/Qwen2.5-7B-Instruct", "vllm-7b",
                          ("--enable-prefix-caching", "--tensor-parallel-size", "2"), gpus=2),
    "1.5b":       Variant("Qwen/Qwen2.5-1.5B-Instruct", "vllm-1.5b", ("--enable-prefix-caching",)),
    # LoRA adapter served under the name `boxer`; adapter path comes from --adapter
    "1.5b-lora":  Variant("Qwen/Qwen2.5-1.5B-Instruct", "vllm-1.5b-lora",
                          ("--enable-prefix-caching", "--enable-lora", "--max-lora-rank", "16")),
}


@dataclass(frozen=True)
class Step:
    tag: str
    kwargs: Tuple[Tuple[str, object], ...] = ()    # RunConfig overrides

    def as_dict(self) -> dict:
        return dict(self.kwargs)


def step(tag: str, **kw) -> Step:
    return Step(tag, tuple(sorted(kw.items())))


PLANS: Dict[str, Dict[str, List[Step]]] = {
    "smoke": {
        "3b": [step("smoke", fights=2, kind="short")],
    },
    # `baseline` (1 fight at a time, short context) is run once per variant and reused by
    # the quantization, caching and batching comparisons in charts.py.
    "core": {
        "3b": [
            step("baseline", concurrency=1),
            step("batching", concurrency=4),
            step("batching", concurrency=16),
            step("caching_full", concurrency=1, context_mode="full"),
            step("capping", kind="long", context_mode="full"),
            step("capping", kind="long", context_mode="capped"),
        ],
        "3b-nocache": [
            step("baseline", concurrency=1),
            step("caching_full", concurrency=1, context_mode="full"),
            step("capping", kind="long", context_mode="full"),
            step("capping", kind="long", context_mode="capped"),
        ],
        "3b-awq": [step("baseline", concurrency=1)],
        "7b-awq": [step("baseline", concurrency=1)],
    },
    "lora": {
        "1.5b": [step("lora", concurrency=1)],
        "1.5b-lora": [step("lora", concurrency=1)],
    },
}


def vllm_command(variant: str, port: int = 8000, adapter: Optional[str] = None,
                 max_model_len: int = 4096, gpu_util: float = 0.90) -> List[str]:
    v = VARIANTS[variant]
    cmd = ["vllm", "serve", v.model, "--port", str(port), "--dtype", "half",
           "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_util),
           *v.flags]
    if "--enable-lora" in v.flags:
        if not adapter:
            raise ValueError(f"variant '{variant}' needs --adapter /path/to/lora_adapter")
        cmd += ["--lora-modules", f"boxer={adapter}"]
    return cmd


def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode()
        return len([l for l in out.splitlines() if l.startswith("GPU")])
    except Exception:
        return 0


def wait_ready(base_url: str, proc: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM not ready after {timeout:.0f}s")


class Server:
    """Context manager: start vLLM, wait until ready, always shut it down."""

    def __init__(self, variant: str, port: int = 8000, adapter: Optional[str] = None):
        self.variant, self.port, self.adapter = variant, port, adapter
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}/v1"

    def __enter__(self):
        # A leftover server (e.g. after an interrupted run) would answer wait_ready() and serve the
        # wrong model, so refuse to start rather than benchmark it by mistake.
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"port {self.port} is already in use (stale vLLM server?). "
                                   "Stop it first: pkill -f 'vllm serve'")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_DIR / f"vllm-{self.variant}.log", "w")
        cmd = vllm_command(self.variant, self.port, self.adapter)
        print(f"[serve] {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        try:
            wait_ready(self.base_url, self.proc)
        except Exception:
            self.__exit__(None, None, None)
            tail = (LOG_DIR / f"vllm-{self.variant}.log").read_text()[-2000:]
            raise RuntimeError(f"server '{self.variant}' failed to start. Log tail:\n{tail}")
        print(f"[serve] {self.variant} ready at {self.base_url}", flush=True)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                self.proc.kill()
        time.sleep(5)    # let the GPU memory free before the next variant


def run_steps(variant: str, steps: List[Step], base_url: str, fights: int, repeats: int,
              extra: Optional[dict] = None):
    from bench.bench import RunConfig, run_bench, print_summary
    v = VARIANTS[variant]
    os.environ["VLLM_BASE_URL"] = base_url
    for s in steps:
        kwargs = dict(backend=v.backend, label=v.label or variant, tag=s.tag,
                      fights=fights, repeats=repeats)
        kwargs.update(s.as_dict())
        kwargs.update(extra or {})
        cfg = RunConfig(**kwargs)
        print(f"\n[bench] {variant} / {s.tag} {s.as_dict()}", flush=True)
        result = run_bench(cfg)
        print_summary(result["summary"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", choices=list(PLANS), default="core")
    p.add_argument("--variants", nargs="*", help="only these variants from the plan")
    p.add_argument("--skip-tags", nargs="*", default=[],
                   help="skip benchmark steps with these tags (e.g. capping: the slow long-fight runs)")
    p.add_argument("--fights", type=int, default=10)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--adapter", help="path to the LoRA adapter (for 1.5b-lora)")
    p.add_argument("--dry-plan", action="store_true", help="print the plan and commands, run nothing")
    p.add_argument("--serve-only", metavar="VARIANT", help="start one server and wait")
    a = p.parse_args(argv)

    if a.serve_only:
        with Server(a.serve_only, a.port, a.adapter) as srv:
            print(f"Serving {a.serve_only} at {srv.base_url}. Ctrl-C to stop.", flush=True)
            try:
                srv.proc.wait()
            except KeyboardInterrupt:
                pass
        return 0

    plan = {k: [s for s in v if s.tag not in a.skip_tags]
            for k, v in PLANS[a.plan].items() if not a.variants or k in a.variants}
    plan = {k: v for k, v in plan.items() if v}
    total_steps = sum(len(v) for v in plan.values())
    print(f"plan '{a.plan}': {len(plan)} servers, {total_steps} benchmark runs, "
          f"{a.fights} fights, {a.repeats} repeat(s)")
    have = gpu_count()
    for name, steps in plan.items():
        v = VARIANTS[name]
        print(f"\n== {name}  ({v.model}, backend {v.backend}, needs {v.gpus} GPU)")
        print("   " + " ".join(vllm_command(name, a.port, a.adapter or "<adapter>")))
        for s in steps:
            print(f"   - {s.tag} {s.as_dict()}")
    if a.dry_plan:
        return 0

    if have == 0:
        print("\nNo GPU found (nvidia-smi). Run this on the Kaggle GPU notebook.", file=sys.stderr)
        return 1
    failed = []
    for name, steps in plan.items():
        v = VARIANTS[name]
        if v.gpus > have:
            print(f"\n[skip] {name} needs {v.gpus} GPUs, found {have}", flush=True)
            continue
        try:
            with Server(name, a.port, a.adapter) as srv:
                run_steps(name, steps, srv.base_url, a.fights, a.repeats)
        except Exception as e:                    # keep going: one bad variant must not lose the rest
            print(f"\n[error] {name}: {e}", file=sys.stderr, flush=True)
            failed.append(name)
    if failed:
        print(f"\nfailed variants: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

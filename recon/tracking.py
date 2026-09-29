"""
Weights & Biases tracking for every Phase 3 training, evaluation and probe run.

One wrapper rather than `wandb` calls scattered through each script, so the
project, run naming and failure behaviour are decided in a single place.

Two rules this encodes:
  * A run that cannot be tracked does not start. Silently producing an
    untracked number is worse than failing, because the number can't be traced
    back to the code and config that made it. `start_run` raises.
  * Tracking is switchable for offline work: `TB_WANDB=0` disables it, and
    every function here accepts the resulting `None` run.

Only config and metrics are logged — never image or CT data.
"""
import os
from pathlib import Path

import wandb

PROJECT = "tb-phase3"
ENV_SWITCH = "TB_WANDB"


class TrackingUnavailable(RuntimeError):
    """W&B could not be reached — stop rather than run untracked."""


def enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "1") not in ("0", "false", "False")


def clean_config(config: dict) -> dict:
    """Paths and other non-JSON values -> strings, so the run config survives."""
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)
    return {k: conv(v) for k, v in config.items()}


def start_run(job_type: str, config: dict, name: str | None = None,
              tags: list[str] | None = None, notes: str | None = None):
    """Begin a tracked run. Returns None only when tracking is switched off."""
    if not enabled():
        print(f"[tracking] {ENV_SWITCH}=0 — running untracked (offline mode)")
        return None
    try:
        run = wandb.init(project=PROJECT, job_type=job_type, name=name,
                         tags=tags, notes=notes, config=clean_config(config))
    except Exception as e:  # auth, network, quota — all fatal here by design
        raise TrackingUnavailable(
            f"W&B unavailable ({e}). Fix authentication/network, or set "
            f"{ENV_SWITCH}=0 to run deliberately untracked.") from e
    print(f"[tracking] {run.url}")
    return run


def log(run, metrics: dict, step: int | None = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def summarize(run, summary: dict) -> None:
    if run is not None:
        run.summary.update(clean_config(summary))


def finish(run, summary: dict | None = None) -> None:
    if run is None:
        return
    if summary:
        summarize(run, summary)
    run.finish()


def run_name(prefix: str, checkpoint: str | Path | None = None) -> str:
    """Names an eval run after the checkpoint it scored, so a reported number
    can be traced to the model that produced it."""
    if checkpoint is None:
        return prefix
    return f"{prefix}-{Path(checkpoint).parent.name}"

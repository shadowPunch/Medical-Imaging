import pytest

from recon import tracking


def test_disabled_by_env_switch_returns_no_run(monkeypatch):
    monkeypatch.setenv(tracking.ENV_SWITCH, "0")
    assert tracking.start_run("train", {"lr": 1e-4}) is None


def test_logging_to_a_disabled_run_is_a_no_op():
    tracking.log(None, {"loss": 1.0}, step=3)
    tracking.finish(None, summary={"psnr": 22.0})


def test_unavailable_wandb_raises_instead_of_running_untracked(monkeypatch):
    """A silently untracked run is worse than a failed one — the whole point of
    mandatory tracking is that results are reproducible afterwards."""
    monkeypatch.delenv(tracking.ENV_SWITCH, raising=False)

    def boom(**kwargs):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(tracking.wandb, "init", boom)
    with pytest.raises(tracking.TrackingUnavailable, match="network unreachable"):
        tracking.start_run("train", {"lr": 1e-4})


def test_config_is_made_json_safe(tmp_path):
    cfg = tracking.clean_config({"out": tmp_path, "lr": 1e-4, "steps": 10, "flag": True,
                                 "nested": {"p": tmp_path / "x.pt"}, "none": None})
    assert cfg["out"] == str(tmp_path)
    assert cfg["nested"]["p"] == str(tmp_path / "x.pt")
    assert cfg["lr"] == 1e-4 and cfg["steps"] == 10 and cfg["flag"] is True
    import json

    json.dumps(cfg)  # must not raise

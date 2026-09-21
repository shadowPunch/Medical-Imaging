import torch
import torch.nn as nn

from recon.train_recon import maybe_resume, save_checkpoint


def _setup():
    model = nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def test_resume_returns_step_one_when_no_checkpoint_exists(tmp_path):
    model, opt = _setup()
    assert maybe_resume(tmp_path / "missing.pt", model, opt, torch.device("cpu")) == 1


def test_resume_continues_from_the_step_after_the_saved_one(tmp_path):
    model, opt = _setup()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, opt, step=400)
    assert maybe_resume(path, model, opt, torch.device("cpu")) == 401


def test_resume_restores_the_saved_weights(tmp_path):
    model, opt = _setup()
    path = tmp_path / "latest.pt"
    with torch.no_grad():
        model.weight.fill_(0.5)
    save_checkpoint(path, model, opt, step=10)

    fresh, fresh_opt = _setup()
    maybe_resume(path, fresh, fresh_opt, torch.device("cpu"))
    assert torch.equal(fresh.weight, torch.full_like(fresh.weight, 0.5))


def test_checkpoint_is_written_atomically(tmp_path):
    """A crash mid-write would otherwise leave a truncated file that kills the
    resume it exists to enable."""
    model, opt = _setup()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, opt, step=1)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))

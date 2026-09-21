import torch

from recon.ct_data import load_ct_volume_cached


class _FakeSubject:
    """Stand-in for a DiffDRR Subject — only needs to survive torch.save/load."""

    def __init__(self, tag):
        self.tag = tag


def test_loads_a_prebuilt_pt_file_directly(tmp_path):
    """The Kaggle/Colab runs ship pre-resampled volumes as .pt files instead of
    DICOM directories, so the loader must accept a file as well as a series dir."""
    pt = tmp_path / "1.2.3.pt"
    torch.save(_FakeSubject("prebuilt"), pt)
    assert load_ct_volume_cached(pt).tag == "prebuilt"


def test_prebuilt_file_is_not_re_cached(tmp_path):
    pt = tmp_path / "1.2.3.pt"
    torch.save(_FakeSubject("prebuilt"), pt)
    load_ct_volume_cached(pt)
    assert not (tmp_path.parent / ".cache").exists()
    assert list(tmp_path.iterdir()) == [pt]

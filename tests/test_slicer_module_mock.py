"""
Executes the Slicer module's code paths against a stubbed Slicer API.

This cannot prove the module works inside real Slicer — only running it there
can, and the stubs necessarily mirror what the code calls rather than what
Slicer actually provides. What it does catch is everything on this side of that
boundary: import errors, wrong argument order, a slider wired to nothing, a
display node never thresholded. Without it the whole UI shell would be
completely unexecuted code.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "slicer_module" / "TBReconstruction.py"


class _FakeDisplayNode:
    def __init__(self):
        self.calls = {}

    def SetApplyThreshold(self, v): self.calls["apply"] = v
    def SetLowerThreshold(self, v): self.calls["lower"] = v
    def SetUpperThreshold(self, v): self.calls["upper"] = v
    def SetAndObserveColorNodeID(self, v): self.calls["color"] = v


class _FakeVolumeNode:
    def __init__(self, path):
        self.path = path
        self.name = None
        self.display = _FakeDisplayNode()

    def SetName(self, name): self.name = name
    def GetDisplayNode(self): return self.display


@pytest.fixture
def slicer_module(monkeypatch):
    """Import TBReconstruction.py with slicer/qt/ctk stubbed out."""
    loaded, layers = [], {}

    util = types.SimpleNamespace(
        loadVolume=lambda p: loaded.append(p) or _FakeVolumeNode(p),
        setSliceViewerLayers=lambda **kw: layers.update(kw),
        arrayFromVolume=lambda node: np.linspace(0, 1, 1000).reshape(10, 10, 10),
    )
    base = types.ModuleType("slicer.ScriptedLoadableModule")
    for name in ("ScriptedLoadableModule", "ScriptedLoadableModuleLogic",
                 "ScriptedLoadableModuleWidget"):
        setattr(base, name, type(name, (), {"__init__": lambda self, *a, **k: None}))

    slicer_stub = types.ModuleType("slicer")
    slicer_stub.util = util
    slicer_stub.mrmlScene = types.SimpleNamespace(Clear=lambda _: None)
    slicer_stub.ScriptedLoadableModule = base

    for name, mod in (("slicer", slicer_stub), ("slicer.ScriptedLoadableModule", base),
                      ("qt", types.ModuleType("qt")), ("ctk", types.ModuleType("ctk"))):
        monkeypatch.setitem(sys.modules, name, mod)

    spec = importlib.util.spec_from_file_location("TBReconstruction_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._loaded, module._layers = loaded, layers
    return module


def _case(tmp_path):
    for name in ("volume.nrrd", "attention.nrrd", "attention_mask.nrrd"):
        (tmp_path / name).write_bytes(b"stub")
    (tmp_path / "report.json").write_text(json.dumps({"tb_probability": 0.87}))
    return tmp_path


def test_load_case_loads_both_volumes_and_the_report(slicer_module, tmp_path):
    volume, attention, report = slicer_module.TBReconstructionLogic().load_case(_case(tmp_path))
    assert [Path(p).name for p in slicer_module._loaded] == ["volume.nrrd", "attention.nrrd"]
    assert report["tb_probability"] == 0.87
    assert "SYNTHESIZED" in volume.name and "SYNTHESIZED" in attention.name


def test_attention_is_shown_over_the_volume_not_instead_of_it(slicer_module, tmp_path):
    volume, attention, _ = slicer_module.TBReconstructionLogic().load_case(_case(tmp_path))
    assert slicer_module._layers["background"] is volume
    assert slicer_module._layers["foreground"] is attention
    assert 0 < slicer_module._layers["foregroundOpacity"] < 1


def test_threshold_is_applied_to_the_display_and_voxels_reported(slicer_module, tmp_path):
    logic = slicer_module.TBReconstructionLogic()
    _, attention, _ = logic.load_case(_case(tmp_path))
    count = logic.apply_threshold(attention, 0.5)
    assert attention.display.calls["apply"] == 1
    assert attention.display.calls["lower"] == 0.5
    assert count == int((np.linspace(0, 1, 1000) >= 0.5).sum())


def test_missing_case_directory_surfaces_a_useful_error(slicer_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="volume.nrrd"):
        slicer_module.TBReconstructionLogic().load_case(tmp_path)

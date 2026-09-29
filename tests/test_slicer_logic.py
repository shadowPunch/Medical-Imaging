import json

import numpy as np
import pytest

from slicer_module.tb_recon_logic import (BANNER, attention_voxels_above, find_outputs,
                                          load_report, probability_summary)


def _case(tmp_path, **report):
    for name in ("volume.nrrd", "attention.nrrd", "attention_mask.nrrd"):
        (tmp_path / name).write_bytes(b"stub")
    (tmp_path / "report.json").write_text(json.dumps(
        {"tb_probability": 0.87, "projection_axis": 2, "threshold": 0.5, **report}))
    return tmp_path


def test_finds_the_pipeline_outputs(tmp_path):
    out = find_outputs(_case(tmp_path))
    assert out["volume"].name == "volume.nrrd"
    assert out["attention"].name == "attention.nrrd"
    assert out["report"].name == "report.json"


def test_missing_output_names_what_is_missing(tmp_path):
    (tmp_path / "volume.nrrd").write_bytes(b"stub")
    with pytest.raises(FileNotFoundError, match="attention.nrrd"):
        find_outputs(tmp_path)


def test_report_round_trips(tmp_path):
    assert load_report(_case(tmp_path))["tb_probability"] == 0.87


def test_probability_summary_states_the_head_it_came_from(tmp_path):
    text = probability_summary(load_report(_case(tmp_path)))
    assert "0.87" in text
    assert "head a" in text.lower() or "diagnostic" in text.lower()


def test_banner_carries_the_limitation_into_the_ui():
    banner = BANNER.lower()
    assert "synthesized" in banner
    assert "not for measurement" in banner


def test_voxel_count_responds_to_the_threshold():
    attention = np.linspace(0, 1, 1000).reshape(10, 10, 10)
    assert attention_voxels_above(attention, 0.0) == 1000
    assert attention_voxels_above(attention, 1.01) == 0
    assert attention_voxels_above(attention, 0.5) < attention_voxels_above(attention, 0.25)

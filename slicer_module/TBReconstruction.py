"""
3D Slicer scripted module: loads the synthesized thoracic volume and Head A's
back-projected localization produced by `recon/export_for_slicer.py`.

Thin by design (proposal §7): inference runs outside Slicer, so Slicer's Python
needs no torch and this file only loads, displays and thresholds. All non-UI
logic lives in `tb_recon_logic.py`, which imports nothing from Slicer and is
unit-tested in this repo's normal test run.

Install: Slicer → Edit → Application Settings → Modules → Additional module
paths → add the directory containing this file, then restart Slicer.
"""
import os
import sys

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleWidget)

sys.path.insert(0, os.path.dirname(__file__))
from tb_recon_logic import BANNER, attention_voxels_above, find_outputs, load_report  # noqa: E402


class TBReconstruction(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "TB CXR 3D Reconstruction"
        parent.categories = ["Chest Imaging"]
        parent.dependencies = []
        parent.contributors = ["TB CXR Diagnostic + 3D Visualization project"]
        parent.helpText = (
            "Loads a synthesized thoracic volume and the diagnostic head's localization "
            "for a chest X-ray, as produced by recon/export_for_slicer.py.\n\n"
            "The volume is an unvalidated visualization aid. The TB probability comes "
            "from the 2D diagnostic head alone and is never derived from the volume."
        )


class TBReconstructionLogic(ScriptedLoadableModuleLogic):
    """Loading and display only — no inference, no diagnostic logic."""

    def load_case(self, directory):
        paths = find_outputs(directory)
        report = load_report(directory)

        slicer.mrmlScene.Clear(0)
        volume = slicer.util.loadVolume(str(paths["volume"]))
        attention = slicer.util.loadVolume(str(paths["attention"]))
        volume.SetName("Synthesized thorax (SYNTHESIZED)")
        attention.SetName("Head A localization (SYNTHESIZED)")

        slicer.util.setSliceViewerLayers(background=volume, foreground=attention,
                                         foregroundOpacity=0.4)
        display = attention.GetDisplayNode()
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
        return volume, attention, report

    def apply_threshold(self, attention_node, threshold):
        """Re-threshold the continuous attention for display, and report how much
        survives — the continuous volume is kept precisely so this stays reversible."""
        display = attention_node.GetDisplayNode()
        display.SetApplyThreshold(1)
        display.SetLowerThreshold(threshold)
        display.SetUpperThreshold(1.0)
        return attention_voxels_above(slicer.util.arrayFromVolume(attention_node), threshold)


class TBReconstructionWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = TBReconstructionLogic()
        self.attention_node = None

        banner = qt.QLabel(BANNER)
        banner.wordWrap = True
        banner.setStyleSheet("QLabel { background-color: #7a1f1f; color: white; padding: 8px; }")
        self.layout.addWidget(banner)

        box = ctk.ctkCollapsibleButton()
        box.text = "Case"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.dir_picker = ctk.ctkPathLineEdit()
        self.dir_picker.filters = ctk.ctkPathLineEdit.Dirs
        form.addRow("Output directory:", self.dir_picker)

        load_button = qt.QPushButton("Load case")
        load_button.connect("clicked(bool)", self.on_load)
        form.addRow(load_button)

        self.probability_label = qt.QLabel("No case loaded.")
        self.probability_label.wordWrap = True
        form.addRow(self.probability_label)

        self.slider = ctk.ctkSliderWidget()
        self.slider.minimum, self.slider.maximum, self.slider.singleStep = 0.0, 1.0, 0.01
        self.slider.value = 0.5
        self.slider.connect("valueChanged(double)", self.on_threshold)
        form.addRow("Attention threshold:", self.slider)

        self.voxel_label = qt.QLabel("")
        form.addRow(self.voxel_label)
        self.layout.addStretch(1)

    def on_load(self):
        try:
            _, self.attention_node, report = self.logic.load_case(self.dir_picker.currentPath)
        except FileNotFoundError as e:
            self.probability_label.text = str(e)
            return
        self.probability_label.text = (
            f"TB probability (Head A, diagnostic model): {report['tb_probability']:.3f}\n"
            "The 3D volume is a visualization aid and played no part in this number.")
        self.on_threshold(self.slider.value)

    def on_threshold(self, value):
        if self.attention_node is None:
            return
        count = self.logic.apply_threshold(self.attention_node, value)
        self.voxel_label.text = f"{count:,} voxels at or above {value:.2f}"

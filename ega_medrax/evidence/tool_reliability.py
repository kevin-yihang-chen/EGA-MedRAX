"""Tool-reliability prior.

A small lookup table that says: "tool t on claim type k typically deserves
weight r(t, k)". Used by the normalizer to rescale raw tool confidence
into a calibrated score before any graph reasoning.

The point of having this as a first-class object is that the verifier can
ingest the prior as input features (so it can learn when to ignore it),
and the training script can update it from validation accuracy.

Defaults encode the intuition from the EGA design doc:
  - classifier strong on FINDING / presence
  - segmentation strong on EXTENT and LOCATION
  - grounding strong on LOCATION
  - VQA broad but noisy
  - report generation broad and noisy on individual claims
  - DICOM metadata authoritative on METADATA
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..schemas import ClaimType


DEFAULT_RELIABILITY: Dict[str, Dict[str, float]] = {
    "chest_xray_classifier": {
        ClaimType.FINDING.value: 0.90,
        ClaimType.DIAGNOSIS.value: 0.60,
        ClaimType.LOCATION.value: 0.20,
        ClaimType.SEVERITY.value: 0.30,
        ClaimType.EXTENT.value: 0.15,
        ClaimType.COMPARISON.value: 0.10,
        ClaimType.DEVICE.value: 0.30,
        ClaimType.METADATA.value: 0.05,
    },
    "chest_xray_segmentation": {
        ClaimType.FINDING.value: 0.55,
        ClaimType.LOCATION.value: 0.85,
        ClaimType.EXTENT.value: 0.90,
        ClaimType.SEVERITY.value: 0.60,
        ClaimType.DIAGNOSIS.value: 0.30,
        ClaimType.COMPARISON.value: 0.20,
        ClaimType.DEVICE.value: 0.20,
        ClaimType.METADATA.value: 0.05,
    },
    "xray_phrase_grounding": {
        ClaimType.LOCATION.value: 0.85,
        ClaimType.FINDING.value: 0.60,
        ClaimType.DEVICE.value: 0.70,
        ClaimType.EXTENT.value: 0.50,
        ClaimType.SEVERITY.value: 0.35,
        ClaimType.DIAGNOSIS.value: 0.30,
        ClaimType.COMPARISON.value: 0.20,
        ClaimType.METADATA.value: 0.05,
    },
    "xray_vqa": {
        ClaimType.FINDING.value: 0.65,
        ClaimType.LOCATION.value: 0.55,
        ClaimType.SEVERITY.value: 0.55,
        ClaimType.DIAGNOSIS.value: 0.55,
        ClaimType.COMPARISON.value: 0.45,
        ClaimType.EXTENT.value: 0.40,
        ClaimType.DEVICE.value: 0.45,
        ClaimType.METADATA.value: 0.20,
    },
    "llava_med": {
        ClaimType.FINDING.value: 0.55,
        ClaimType.LOCATION.value: 0.50,
        ClaimType.SEVERITY.value: 0.50,
        ClaimType.DIAGNOSIS.value: 0.60,
        ClaimType.COMPARISON.value: 0.45,
        ClaimType.EXTENT.value: 0.35,
        ClaimType.DEVICE.value: 0.40,
        ClaimType.METADATA.value: 0.20,
    },
    "chest_xray_report_generator": {
        ClaimType.FINDING.value: 0.55,
        ClaimType.LOCATION.value: 0.45,
        ClaimType.SEVERITY.value: 0.45,
        ClaimType.DIAGNOSIS.value: 0.55,
        ClaimType.COMPARISON.value: 0.35,
        ClaimType.EXTENT.value: 0.30,
        ClaimType.DEVICE.value: 0.40,
        ClaimType.METADATA.value: 0.10,
    },
    "dicom_processor": {
        ClaimType.METADATA.value: 0.98,
        ClaimType.FINDING.value: 0.10,
    },
}


class ToolReliabilityPrior:
    """Encapsulates the r(tool, claim_type) table.

    The lookup degrades gracefully: an unknown (tool, claim_type) pair
    returns the configured default rather than raising. This matters
    because tool names vary between MedRAX deployments and we want the
    graph to keep functioning even when we cannot recognise a tool.
    """

    def __init__(self, table: Optional[Dict[str, Dict[str, float]]] = None, default: float = 0.4):
        self.table = table if table is not None else {k: dict(v) for k, v in DEFAULT_RELIABILITY.items()}
        self.default = default

    def weight(self, tool_name: str, claim_type: ClaimType) -> float:
        per_tool = self.table.get(tool_name)
        if per_tool is None:
            return self.default
        return per_tool.get(claim_type.value, self.default)

    def calibrate(self, raw_score: float, tool_name: str, claim_type: ClaimType) -> float:
        """Rescale a raw [0, 1] score by the tool reliability for that claim type."""
        w = self.weight(tool_name, claim_type)
        return float(max(0.0, min(1.0, raw_score * w)))

    # --- I/O ----------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"table": self.table, "default": self.default}, indent=2))

    @classmethod
    def load(cls, path: str) -> "ToolReliabilityPrior":
        data = json.loads(Path(path).read_text())
        return cls(table=data["table"], default=data["default"])

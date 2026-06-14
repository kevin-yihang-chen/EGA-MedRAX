"""Module C - Evidence Normalizer.

Every tool in MedRAX returns a different shape (a dict of probabilities, a
free-text VQA answer, a mask, a bbox + phrase, ...). The normalizer is the
single chokepoint that converts those heterogeneous shapes into typed
`Evidence` objects bound to a target Claim.

Why this is the keystone module:

  * The graph verifier never sees raw tool output - it only sees
    normalized propositions (SUPPORTS / CONTRADICTS / INSUFFICIENT) plus
    a calibrated score and optional region anchor. That means a new tool
    can be added without touching the verifier.
  * Contradiction is detected here, not later: if a classifier returns
    p=0.1 for "Effusion" against a positive claim, the normalizer emits
    a CONTRADICTS edge with score 0.9 rather than dropping the result.
  * The mapping from tool name to handler is data-driven; users register
    custom tools at runtime.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..schemas import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceType,
    Polarity,
    Proposition,
    Region,
)
from .tool_reliability import ToolReliabilityPrior


# Canonical synonyms that map free user phrasing onto the classifier vocabulary.
_PATHOLOGY_ALIASES: Dict[str, str] = {
    "pleural effusion": "Effusion",
    "effusion": "Effusion",
    "fluid in lung": "Effusion",
    "cardiomegaly": "Cardiomegaly",
    "enlarged heart": "Cardiomegaly",
    "consolidation": "Consolidation",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "ptx": "Pneumothorax",
    "edema": "Edema",
    "atelectasis": "Atelectasis",
    "nodule": "Nodule",
    "mass": "Mass",
    "opacity": "Lung Opacity",
    "lung opacity": "Lung Opacity",
    "fracture": "Fracture",
}

# Threshold above which a presence-probability is treated as a positive call.
PRESENCE_TAU = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_pathology(subject: str) -> str:
    key = subject.strip().lower()
    return _PATHOLOGY_ALIASES.get(key, subject.title())


def _affirmative(text: str) -> bool:
    """Heuristic for a yes/no style answer."""
    if not text:
        return False
    t = text.strip().lower()
    if re.match(r"^(yes|yeah|y\b|present|positive|likely|probable|consistent)\b", t):
        return True
    if re.search(r"\b(no|not|negative|absent|unlikely|denies|without)\b", t[:40]):
        return False
    return True  # default: free-form description usually describes a finding


def _polarity_aligns(p: Polarity, observed_present: bool) -> Proposition:
    """Decide whether an observation supports or contradicts the claim."""
    if p == Polarity.POSITIVE:
        return Proposition.SUPPORTS if observed_present else Proposition.CONTRADICTS
    if p == Polarity.NEGATIVE:
        return Proposition.CONTRADICTS if observed_present else Proposition.SUPPORTS
    return Proposition.INSUFFICIENT


# ---------------------------------------------------------------------------
# Per-tool handlers
# ---------------------------------------------------------------------------


HandlerOut = Tuple[List[Evidence], List[Region]]
Handler = Callable[["EvidenceNormalizer", Claim, Any, Dict[str, Any]], HandlerOut]


class EvidenceNormalizer:
    """Convert raw tool outputs into Evidence/Region objects bound to a Claim."""

    def __init__(self, reliability: Optional[ToolReliabilityPrior] = None):
        self.reliability = reliability or ToolReliabilityPrior()
        self._handlers: Dict[str, Handler] = {
            "chest_xray_classifier": _handle_classifier,
            "chest_xray_segmentation": _handle_segmentation,
            "xray_phrase_grounding": _handle_grounding,
            "xray_vqa": _handle_vqa,
            "llava_med": _handle_vqa,
            "chest_xray_report_generator": _handle_report,
            "dicom_processor": _handle_dicom,
        }

    def register(self, tool_name: str, handler: Handler) -> None:
        self._handlers[tool_name] = handler

    def normalize(
        self,
        tool_name: str,
        raw_output: Any,
        claim: Claim,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> HandlerOut:
        """Dispatch to a per-tool handler.

        Unknown tools fall back to a generic text handler so the pipeline
        keeps running; they just produce a low-weight INSUFFICIENT edge.
        """
        handler = self._handlers.get(tool_name, _handle_unknown)
        provenance = dict(provenance or {})
        provenance.setdefault("tool", tool_name)
        return handler(self, claim, raw_output, provenance)

    # --- public helpers used by handlers ---------------------------------

    def make_evidence(
        self,
        claim: Claim,
        tool_name: str,
        evidence_type: EvidenceType,
        proposition: Proposition,
        score: float,
        *,
        uncertainty: float = 0.0,
        region_id: Optional[str] = None,
        text: str = "",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Evidence:
        calibrated = self.reliability.calibrate(score, tool_name, claim.type)
        return Evidence(
            claim_id=claim.id,
            tool_name=tool_name,
            evidence_type=evidence_type,
            proposition=proposition,
            score=float(score),
            calibrated_score=float(calibrated),
            uncertainty=float(uncertainty),
            region_id=region_id,
            text=text,
            provenance=provenance or {},
        )


# ---------------------------------------------------------------------------
# Tool-specific handlers
# ---------------------------------------------------------------------------


def _handle_classifier(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """Convert {pathology: prob} predictions into a single Evidence per claim."""
    payload = _unwrap_tool_output(raw)
    if not isinstance(payload, dict) or "error" in payload:
        return [self.make_evidence(
            claim, "chest_xray_classifier", EvidenceType.CLASSIFICATION,
            Proposition.INSUFFICIENT, score=0.0, text="classifier error",
            provenance=prov,
        )], []
    target = _canonical_pathology(claim.subject)
    if target not in payload:
        return [self.make_evidence(
            claim, "chest_xray_classifier", EvidenceType.CLASSIFICATION,
            Proposition.NOT_APPLICABLE, score=0.0,
            text=f"no probability for {target}", provenance=prov,
        )], []
    prob = float(payload[target])
    present = prob >= PRESENCE_TAU
    prop = _polarity_aligns(claim.polarity, present)
    score = prob if present else (1.0 - prob)
    return [self.make_evidence(
        claim, "chest_xray_classifier", EvidenceType.CLASSIFICATION,
        prop, score=score, uncertainty=_binary_entropy(prob),
        text=f"P({target})={prob:.3f}", provenance={**prov, "p": prob, "target": target},
    )], []


def _handle_segmentation(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """Segmentation supports presence and localization.

    We accept either:
      - {"masks": {label: {"area": float, "side": "left", ...}}}
      - {"<label>": area_fraction}
    """
    payload = _unwrap_tool_output(raw)
    if not isinstance(payload, dict):
        return [], []
    masks = payload.get("masks", payload)
    target = _canonical_pathology(claim.subject).lower()
    best_label: Optional[str] = None
    best_info: Dict[str, Any] = {}
    best_area = 0.0
    for label, info in masks.items():
        if not isinstance(label, str):
            continue
        if target in label.lower() or label.lower() in target:
            if isinstance(info, dict):
                area = float(info.get("area", info.get("area_fraction", 0.0)))
            else:
                area = float(info)
                info = {"area": area}
            if area >= best_area:
                best_area = area
                best_label = label
                best_info = info
    if best_label is None:
        return [self.make_evidence(
            claim, "chest_xray_segmentation", EvidenceType.SEGMENTATION,
            Proposition.NOT_APPLICABLE, score=0.0,
            text="no matching mask", provenance=prov,
        )], []

    region = Region(
        label=best_label,
        side=best_info.get("side"),
        zone=best_info.get("zone"),
        bbox=tuple(best_info["bbox"]) if isinstance(best_info.get("bbox"), (list, tuple)) else None,
        mask_ref=best_info.get("mask_ref"),
    )
    present = best_area > 0.0
    prop = _polarity_aligns(claim.polarity, present)
    score = min(1.0, best_area * 10.0) if present else 0.5
    if claim.anatomy and region.side and region.side not in claim.anatomy:
        prop = Proposition.CONTRADICTS
        score = max(score, 0.6)
    return [self.make_evidence(
        claim, "chest_xray_segmentation", EvidenceType.SEGMENTATION,
        prop, score=score, region_id=region.id,
        text=f"mask area={best_area:.3f} side={region.side}", provenance=prov,
    )], [region]


def _handle_grounding(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """Phrase grounding returns {phrase: bbox, score}. We pick the best matching phrase."""
    payload = _unwrap_tool_output(raw)
    if not isinstance(payload, dict):
        return [], []
    boxes = payload.get("boxes", payload)
    target = _canonical_pathology(claim.subject).lower()
    best_score = 0.0
    best_box = None
    best_phrase = ""
    for phrase, info in boxes.items() if isinstance(boxes, dict) else []:
        if not isinstance(phrase, str):
            continue
        if target not in phrase.lower():
            continue
        if isinstance(info, dict):
            score = float(info.get("score", info.get("confidence", 0.5)))
            bbox = info.get("bbox")
        else:
            score = 0.5
            bbox = info
        if score > best_score:
            best_score = score
            best_box = bbox
            best_phrase = phrase

    if best_box is None:
        return [self.make_evidence(
            claim, "xray_phrase_grounding", EvidenceType.GROUNDING,
            Proposition.NOT_APPLICABLE, score=0.0,
            text="no phrase match", provenance=prov,
        )], []

    bbox = tuple(best_box) if isinstance(best_box, (list, tuple)) and len(best_box) == 4 else None
    side = _side_from_bbox(bbox)
    region = Region(label=best_phrase, side=side, bbox=bbox)
    present = True
    prop = _polarity_aligns(claim.polarity, present)
    if claim.type == ClaimType.LOCATION and claim.attribute and side and side != claim.attribute:
        prop = Proposition.CONTRADICTS
    return [self.make_evidence(
        claim, "xray_phrase_grounding", EvidenceType.GROUNDING,
        prop, score=best_score, region_id=region.id,
        text=f"{best_phrase} @ {side or 'unknown'}", provenance=prov,
    )], [region]


def _handle_vqa(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """VQA tools return free text - the most hallucination-prone source."""
    payload = _unwrap_tool_output(raw)
    if isinstance(payload, dict):
        text = str(payload.get("answer", payload.get("text", payload)))
    else:
        text = str(payload)
    target = _canonical_pathology(claim.subject).lower()
    mentions = target in text.lower() or claim.subject.lower() in text.lower()
    if not mentions:
        return [self.make_evidence(
            claim, prov.get("tool", "xray_vqa"), EvidenceType.VQA,
            Proposition.INSUFFICIENT, score=0.0, text=text[:200], provenance=prov,
        )], []
    affirmative = _affirmative(text)
    prop = _polarity_aligns(claim.polarity, affirmative)
    score = 0.7 if affirmative else 0.6
    return [self.make_evidence(
        claim, prov.get("tool", "xray_vqa"), EvidenceType.VQA,
        prop, score=score, uncertainty=0.3, text=text[:200], provenance=prov,
    )], []


def _handle_report(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """Treat a generated report as a single noisy text source per claim."""
    payload = _unwrap_tool_output(raw)
    text = payload if isinstance(payload, str) else str(payload.get("report", payload))
    target = _canonical_pathology(claim.subject).lower()
    mentions = target in text.lower()
    if not mentions:
        return [self.make_evidence(
            claim, "chest_xray_report_generator", EvidenceType.REPORT,
            Proposition.INSUFFICIENT, score=0.0, text=text[:200], provenance=prov,
        )], []
    affirmative = _affirmative(_window(text, target, 80))
    prop = _polarity_aligns(claim.polarity, affirmative)
    return [self.make_evidence(
        claim, "chest_xray_report_generator", EvidenceType.REPORT,
        prop, score=0.6, uncertainty=0.35, text=_window(text, target, 80),
        provenance=prov,
    )], []


def _handle_dicom(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """DICOM metadata is most useful for METADATA and COMPARISON claims."""
    payload = _unwrap_tool_output(raw)
    if not isinstance(payload, dict):
        return [], []
    return [self.make_evidence(
        claim, "dicom_processor", EvidenceType.METADATA,
        Proposition.SUPPORTS if claim.type == ClaimType.METADATA else Proposition.INSUFFICIENT,
        score=0.9 if claim.type == ClaimType.METADATA else 0.1,
        text=str({k: payload.get(k) for k in ("StudyDate", "PixelSpacing", "ViewPosition") if k in payload})[:200],
        provenance={**prov, **{k: payload.get(k) for k in payload if not k.startswith("_")}},
    )], []


def _handle_unknown(self: EvidenceNormalizer, claim: Claim, raw: Any, prov: Dict[str, Any]) -> HandlerOut:
    """Last-resort handler.

    Captures the tool output as text and emits an INSUFFICIENT edge so
    the verifier knows about the call but does not weigh it heavily.
    """
    text = str(raw)[:300]
    return [self.make_evidence(
        claim, prov.get("tool", "unknown"), EvidenceType.RETRIEVED_TEXT,
        Proposition.INSUFFICIENT, score=0.0, uncertainty=0.5, text=text, provenance=prov,
    )], []


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _unwrap_tool_output(raw: Any) -> Any:
    """MedRAX tools return (output, metadata) tuples; transparently unwrap."""
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw[0]
    return raw


def _binary_entropy(p: float) -> float:
    import math
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2))


def _side_from_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[str]:
    if bbox is None:
        return None
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    # bbox normalised to [0, 1]; radiological convention: image left = patient right
    if cx < 0.45:
        return "right"
    if cx > 0.55:
        return "left"
    return "midline"


def _window(text: str, needle: str, span: int) -> str:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return text[:span]
    start = max(0, idx - span // 2)
    end = min(len(text), idx + span)
    return text[start:end]

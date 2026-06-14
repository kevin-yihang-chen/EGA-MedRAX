"""Module B - Evidence Collector.

Given a Claim and a tool registry, decide which tools to call (initial
acquisition) or which extra tools to call (adaptive refinement), invoke
them on the input image, and push the normalized Evidence/Region into the
graph.

Design notes:
  * The collector does NOT freely choose tools - it uses a
    type -> [tool_names] routing table so the trajectory is auditable.
    The router is a single small dict that can be swapped out (e.g. for
    a learned router in the version-2 method).
  * Each tool call is wrapped in a try/except and turned into a NOT_APPLICABLE
    or INSUFFICIENT evidence node on failure, so a broken tool degrades
    the verifier's input rather than crashing the agent.
  * `refine` is for the verifier-driven feedback loop (step 5 of inference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from ..schemas import Claim, ClaimType, EvidenceGraph, VerifierOutput
from .normalizer import EvidenceNormalizer


logger = logging.getLogger(__name__)


# Initial tool routing per claim type.
DEFAULT_INITIAL_ROUTING: Dict[ClaimType, List[str]] = {
    ClaimType.FINDING: ["chest_xray_classifier", "xray_vqa"],
    ClaimType.LOCATION: ["chest_xray_segmentation", "xray_phrase_grounding"],
    ClaimType.SEVERITY: ["chest_xray_classifier", "chest_xray_segmentation"],
    ClaimType.EXTENT: ["chest_xray_segmentation"],
    ClaimType.COMPARISON: ["xray_vqa", "chest_xray_report_generator"],
    ClaimType.DIAGNOSIS: ["xray_vqa", "chest_xray_report_generator", "chest_xray_classifier"],
    ClaimType.DEVICE: ["xray_phrase_grounding", "xray_vqa"],
    ClaimType.METADATA: ["dicom_processor"],
}

# Adaptive routing - which tool to call when the verifier flags a gap.
DEFAULT_REFINEMENT_ROUTING: Dict[str, List[str]] = {
    "insufficient": ["llava_med", "xray_phrase_grounding"],
    "conflict": ["xray_vqa", "llava_med"],
    "ungrounded": ["xray_phrase_grounding", "chest_xray_segmentation"],
}


@dataclass
class CollectorConfig:
    max_tools_per_claim: int = 3
    max_refinement_passes: int = 2
    initial_routing: Dict[ClaimType, List[str]] = field(default_factory=lambda: dict(DEFAULT_INITIAL_ROUTING))
    refinement_routing: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_REFINEMENT_ROUTING))
    sufficiency_threshold: float = 0.55
    conflict_threshold: float = 0.4


class EvidenceCollector:
    """Routes claims to tools, calls them, and merges the result into a graph.

    Parameters
    ----------
    tools : dict[str, BaseTool-like]
        Mapping from tool name to an object exposing ``invoke(args)`` (the
        MedRAX BaseTool interface). The collector is therefore drop-in
        compatible with the existing MedRAX tool ecosystem.
    normalizer : EvidenceNormalizer
        Converts raw tool output into Evidence / Region.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        normalizer: EvidenceNormalizer,
        config: Optional[CollectorConfig] = None,
    ):
        self.tools = tools
        self.normalizer = normalizer
        self.config = config or CollectorConfig()
        self._call_log: List[Dict[str, Any]] = []

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        return list(self._call_log)

    # ------------------------------------------------------------------
    # Initial pass
    # ------------------------------------------------------------------

    def collect_initial(
        self,
        claims: List[Claim],
        image_path: str,
        graph: EvidenceGraph,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> EvidenceGraph:
        """Acquire a first round of evidence for every claim."""
        extra = extra_inputs or {}
        for claim in claims:
            graph.add_claim(claim)
            routed = self.config.initial_routing.get(claim.type, ["chest_xray_classifier"])
            # also fan out to parent-claim-relevant tools so diagnosis claims
            # see classifier output for their finding parents.
            self._call_tools_for_claim(claim, routed[: self.config.max_tools_per_claim],
                                       image_path, graph, extra)
        for claim in claims:
            for parent_id in claim.parents:
                graph.add_relation(parent_id, claim.id, "implies")
        return graph

    # ------------------------------------------------------------------
    # Adaptive refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        verifier_outputs: Dict[str, VerifierOutput],
        image_path: str,
        graph: EvidenceGraph,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Call extra tools for claims the verifier flagged as gappy.

        Returns the ids of the claims that were refined - the agent loop
        uses this to decide whether another verification pass is needed.
        """
        extra = extra_inputs or {}
        refined: List[str] = []
        for claim_id, vout in verifier_outputs.items():
            claim = graph.claims.get(claim_id)
            if claim is None:
                continue
            tools_to_call = self._pick_refinement_tools(claim, vout, graph)
            if not tools_to_call:
                continue
            self._call_tools_for_claim(claim, tools_to_call, image_path, graph, extra)
            refined.append(claim_id)
        return refined

    def _pick_refinement_tools(
        self, claim: Claim, vout: VerifierOutput, graph: EvidenceGraph,
    ) -> List[str]:
        already_called: Set[str] = {ev.tool_name for ev in graph.evidence_for(claim.id)}
        candidates: List[str] = []
        if vout.sufficiency < self.config.sufficiency_threshold:
            candidates.extend(self.config.refinement_routing.get("insufficient", []))
        if vout.conflict > self.config.conflict_threshold:
            candidates.extend(self.config.refinement_routing.get("conflict", []))
        if not any(ev.region_id for ev in graph.evidence_for(claim.id)):
            candidates.extend(self.config.refinement_routing.get("ungrounded", []))
        picked: List[str] = []
        for t in candidates:
            if t in already_called or t not in self.tools:
                continue
            picked.append(t)
            if len(picked) >= self.config.max_tools_per_claim:
                break
        return picked

    # ------------------------------------------------------------------
    # Internal: actually invoke tools
    # ------------------------------------------------------------------

    def _call_tools_for_claim(
        self,
        claim: Claim,
        tool_names: List[str],
        image_path: str,
        graph: EvidenceGraph,
        extra: Dict[str, Any],
    ) -> None:
        for tool_name in tool_names:
            tool = self.tools.get(tool_name)
            if tool is None:
                continue
            try:
                args = self._build_args(tool_name, claim, image_path, extra)
                raw = self._invoke(tool, args)
                status = "ok"
            except Exception as e:  # noqa: BLE001
                logger.warning("tool %s failed for claim %s: %s", tool_name, claim.id, e)
                raw = {"error": str(e)}
                status = "error"

            self._call_log.append({
                "claim_id": claim.id, "tool_name": tool_name,
                "args": args, "status": status,
            })
            evidence, regions = self.normalizer.normalize(
                tool_name, raw, claim, provenance={"args": args, "status": status},
            )
            for r in regions:
                graph.add_region(r)
            for ev in evidence:
                graph.add_evidence(ev)

    def _build_args(self, tool_name: str, claim: Claim, image_path: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Translate (tool, claim, image) into the kwargs each tool expects.

        Different MedRAX tools take slightly different arg names; we
        centralise that mapping here so the rest of the pipeline only
        talks about claims and images.
        """
        if tool_name in ("xray_vqa", "llava_med"):
            return {"image_path": image_path, "question": _claim_to_question(claim)}
        if tool_name == "xray_phrase_grounding":
            return {"image_path": image_path, "phrase": claim.subject or claim.text}
        return {"image_path": image_path}

    @staticmethod
    def _invoke(tool: Any, args: Dict[str, Any]) -> Any:
        """MedRAX tools follow the langchain BaseTool ``invoke`` interface."""
        if hasattr(tool, "invoke"):
            return tool.invoke(args)
        if callable(tool):
            return tool(**args)
        raise TypeError(f"tool {tool!r} has no invoke method and is not callable")


# ---------------------------------------------------------------------------
# Question templates used to query VQA-style tools per claim
# ---------------------------------------------------------------------------


def _claim_to_question(claim: Claim) -> str:
    base = claim.text.strip().rstrip(".?!")
    if claim.type == ClaimType.FINDING:
        side = f" on the {claim.anatomy}" if claim.anatomy else ""
        return f"Is there {claim.subject}{side} in this chest X-ray? Answer yes or no, then briefly justify."
    if claim.type == ClaimType.LOCATION:
        return f"Where is the {claim.subject} located in this chest X-ray? Left, right, bilateral, or absent?"
    if claim.type == ClaimType.SEVERITY:
        return f"How severe is the {claim.subject}? Mild, moderate, severe, or absent?"
    if claim.type == ClaimType.COMPARISON:
        return f"Has the {claim.subject} changed compared with the prior study? Worsened, improved, or unchanged?"
    if claim.type == ClaimType.DIAGNOSIS:
        return f"Is the diagnosis '{base}' supported by this chest X-ray? Answer yes or no, then briefly justify."
    if claim.type == ClaimType.DEVICE:
        return f"Is the {claim.subject} positioned correctly in this chest X-ray?"
    return f"{base}?"

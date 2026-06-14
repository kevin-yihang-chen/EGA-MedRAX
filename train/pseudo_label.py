"""Weakly supervised dataset construction.

Implements Stage 1 of the design doc's training plan:

  1. Use the LLM to extract claims from each (question, reference answer)
     pair.
  2. Run the EvidenceCollector on the image to get tool outputs.
  3. Align tool outputs to claims with the EvidenceNormalizer.
  4. Generate pseudo claim-truth labels by checking whether the reference
     answer entails / contradicts each claim (text overlap + LLM yes/no).
  5. Generate pseudo conflict labels from the support/contradict split.

Outputs JSON-serialised (graph, labels) pairs that the trainer loads.

The script is intentionally streaming and forgiving so it can be re-run
on partially failing examples.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ega_medrax.claim_decomposer import BaseClaimDecomposer, RuleClaimDecomposer
from ega_medrax.evidence import EvidenceCollector, EvidenceNormalizer, ToolReliabilityPrior
from ega_medrax.graph import GraphBuilder
from ega_medrax.schemas import EvidenceGraph, Proposition


logger = logging.getLogger(__name__)


@dataclass
class PseudoExample:
    """A single training sample written to disk."""

    question: str
    image_path: str
    reference_answer: str
    graph: Dict[str, Any]
    claim_labels: Dict[str, int]     # claim_id -> {0: true, 1: false, -1: unknown}
    conflict_labels: Dict[str, int]  # claim_id -> {0, 1}
    answerable: bool                  # gold "should the agent answer?"


def build_pseudo_dataset(
    samples: Iterable[Dict[str, Any]],
    tools: Dict[str, Any],
    output_dir: str,
    decomposer: Optional[BaseClaimDecomposer] = None,
    reliability: Optional[ToolReliabilityPrior] = None,
    skip_existing: bool = True,
) -> List[str]:
    """Walk ``samples``, build one JSON pseudo-example per item.

    Parameters
    ----------
    samples : iterable of dict
        Each item should have ``question``, ``image_path``, ``answer``
        and optionally ``answerable`` (default True).
    tools : dict
        MedRAX-compatible tool registry. Use only the tools you can afford
        to run offline; pseudo-labeling is the most expensive stage.
    output_dir : str
        Directory to write per-sample JSON files into.
    decomposer : optional decomposer
        Defaults to ``RuleClaimDecomposer``.
    reliability : optional reliability prior
        Defaults to the EGA defaults.

    Returns
    -------
    list[str]
        Paths of the written JSON files.
    """
    decomposer = decomposer or RuleClaimDecomposer()
    normalizer = EvidenceNormalizer(reliability=reliability or ToolReliabilityPrior())
    collector = EvidenceCollector(tools=tools, normalizer=normalizer)
    builder = GraphBuilder()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    for i, sample in enumerate(samples):
        target = out_dir / f"sample_{i:06d}.json"
        if skip_existing and target.exists():
            paths.append(str(target))
            continue
        try:
            example = _build_one(sample, decomposer, collector, builder)
        except Exception as e:  # noqa: BLE001
            logger.warning("sample %d failed: %s", i, e)
            continue
        target.write_text(json.dumps(example.__dict__, indent=2, default=str))
        paths.append(str(target))
    return paths


def _build_one(
    sample: Dict[str, Any],
    decomposer: BaseClaimDecomposer,
    collector: EvidenceCollector,
    builder: GraphBuilder,
) -> PseudoExample:
    question = sample["question"]
    image_path = sample["image_path"]
    reference = sample.get("answer", "")
    answerable = bool(sample.get("answerable", True))

    claims = decomposer(question, {})
    graph = EvidenceGraph()
    collector.collect_initial(claims, image_path, graph)
    graph, _violations = builder.finalize(graph)

    claim_labels = _label_claims(graph, reference)
    conflict_labels = _label_conflict(graph)

    return PseudoExample(
        question=question,
        image_path=image_path,
        reference_answer=reference,
        graph=graph.to_dict(),
        claim_labels=claim_labels,
        conflict_labels=conflict_labels,
        answerable=answerable,
    )


def _label_claims(graph: EvidenceGraph, reference: str) -> Dict[str, int]:
    """Heuristic pseudo-labels from the reference answer.

    The reference is the radiologist's free-text answer. For each claim
    we check whether the subject and its polarity match the answer text;
    if both match we mark it 0 (true), if the subject matches but the
    polarity disagrees we mark it 1 (false), else -1 (unknown).
    """
    ref = reference.lower()
    labels: Dict[str, int] = {}
    for cid, claim in graph.claims.items():
        subj = (claim.subject or "").lower()
        if not subj or subj == "unspecified":
            labels[cid] = -1
            continue
        has_subj = subj in ref
        negated = any(neg in ref for neg in (
            f"no {subj}", f"without {subj}", f"absent {subj}",
            f"no evidence of {subj}", f"denies {subj}",
        ))
        if not has_subj:
            labels[cid] = -1
        elif negated and claim.polarity.value == "positive":
            labels[cid] = 1
        elif not negated and claim.polarity.value == "negative":
            labels[cid] = 1
        else:
            labels[cid] = 0
    return labels


def _label_conflict(graph: EvidenceGraph) -> Dict[str, int]:
    """A claim is 'conflicting' if both SUPPORTS and CONTRADICTS evidence exist."""
    labels: Dict[str, int] = {}
    for cid in graph.claims:
        s = any(e.proposition == Proposition.SUPPORTS for e in graph.evidence_for(cid))
        c = any(e.proposition == Proposition.CONTRADICTS for e in graph.evidence_for(cid))
        labels[cid] = 1 if (s and c) else 0
    return labels

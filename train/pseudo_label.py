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

CLI
---
The module is runnable as ``python -m train.pseudo_label``. Provide a
JSONL file of samples (one ``{"question", "image_path", "answer", ...}``
per line) and an output directory. The tool registry is constructed by
the helper named after ``--tool_loader``; the default ``mock`` uses
``examples.mock_tools`` so users can dry-run the pipeline. Real runs pass
``--tool_loader medrax`` to use ``main.build_agent``'s MedRAX factories.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_tools_mock(case: str) -> Dict[str, Any]:
    """Tool loader for dry-runs - uses examples/mock_tools.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.mock_tools import build_mock_tools  # type: ignore
    return build_mock_tools(case)


def _load_tools_factory(spec: str) -> Dict[str, Any]:
    """Resolve ``module:function`` and call it with no arguments.

    Used to inject a real MedRAX tool registry without baking a hard
    dependency into this module. Example:
        --tool_loader my_pkg.tool_factories:build_tools
    """
    if ":" not in spec:
        raise ValueError(f"--tool_loader expects 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn: Callable[[], Dict[str, Any]] = getattr(module, fn_name)
    return fn()


def _resolve_decomposer(name: str, llm: Optional[Any]) -> BaseClaimDecomposer:
    if name == "rule":
        return RuleClaimDecomposer()
    if name == "llm":
        from ega_medrax.claim_decomposer import LLMClaimDecomposer
        if llm is None:
            raise ValueError("--decomposer llm requires --llm_model")
        return LLMClaimDecomposer(llm)
    raise ValueError(f"unknown decomposer: {name}")


def _build_llm(model: str) -> Any:
    import os
    from langchain_openai import ChatOpenAI
    kwargs: Dict[str, Any] = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    return ChatOpenAI(model=model, temperature=0.0, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a weakly supervised EGA graph dataset.",
    )
    parser.add_argument(
        "--in", dest="input", required=True,
        help="JSONL file of {question, image_path, answer, [answerable]} samples.",
    )
    parser.add_argument(
        "--out", dest="output", required=True,
        help="Output directory for per-sample JSON pseudo-examples.",
    )
    parser.add_argument(
        "--tool_loader", default="mock",
        help=(
            "Either 'mock' (uses examples/mock_tools.py) or a 'module:function' "
            "spec that returns the tool registry dict."
        ),
    )
    parser.add_argument(
        "--mock_case", default="left_effusion",
        help="Case key for the mock loader when --tool_loader=mock.",
    )
    parser.add_argument(
        "--decomposer", choices=["rule", "llm"], default="rule",
        help="Claim decomposer: 'rule' (offline) or 'llm' (requires --llm_model).",
    )
    parser.add_argument(
        "--llm_model", default=None,
        help="OpenAI-compatible model name; only used when --decomposer=llm.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip samples whose output file already exists (default: True).",
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-run all samples, overwriting existing outputs.",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.tool_loader == "mock":
        tools = _load_tools_mock(args.mock_case)
    else:
        tools = _load_tools_factory(args.tool_loader)

    llm = _build_llm(args.llm_model) if args.llm_model else None
    decomposer = _resolve_decomposer(args.decomposer, llm)

    samples = list(_iter_jsonl(args.input))
    logger.info("loaded %d samples from %s", len(samples), args.input)

    paths = build_pseudo_dataset(
        samples=samples,
        tools=tools,
        output_dir=args.output,
        decomposer=decomposer,
        skip_existing=args.skip_existing,
    )
    logger.info("wrote %d pseudo-examples to %s", len(paths), args.output)


if __name__ == "__main__":
    main()

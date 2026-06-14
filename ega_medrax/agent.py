"""EGAAgent - the orchestrator.

Re-implements the MedRAX agent loop as a six-node LangGraph workflow:

    query
     -> decompose_claims
     -> collect_evidence
     -> build_graph
     -> verify_graph
     -> [enough?]
          yes -> answer/abstain
          no  -> execute_more_tools -> build_graph -> verify_graph

The agent is designed to be drop-in compatible with the MedRAX tool
registry, so users can take their existing ``initialize_agent`` setup,
swap ``Agent`` for ``EGAAgent``, and get the structured pipeline.
"""

from __future__ import annotations

import json
import logging
import operator
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
except ImportError as e:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None  # type: ignore
    _LANGGRAPH_ERR = e
else:
    _LANGGRAPH_ERR = None

from .schemas import (
    AbstentionDecision,
    AbstentionReason,
    Claim,
    EvidenceGraph,
    VerifierOutput,
)
from .claim_decomposer import (
    BaseClaimDecomposer,
    LLMClaimDecomposer,
    RuleClaimDecomposer,
)
from .evidence import EvidenceCollector, EvidenceNormalizer, ToolReliabilityPrior
from .graph import GraphBuilder, RuleViolation
from .verifier import BaseVerifier, HybridVerifier, RuleVerifier
from .abstention import AbstentionPolicy
from .abstention.answer_head import AnswerHead, TemplateAnswerHead


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class EGAState(TypedDict, total=False):
    """LangGraph state.

    Compared with MedRAX's plain ``messages``-only AgentState we carry
    the entire evidence reasoning context as first-class fields. This is
    the structural change called out in section "8. 和 MedRAX 代码怎么对接"
    of the design doc.
    """

    messages: Annotated[List[Any], operator.add]
    question: str
    image_path: str
    extra_inputs: Dict[str, Any]
    claims: List[Claim]
    graph: EvidenceGraph
    violations: List[RuleViolation]
    verifier_outputs: Dict[str, VerifierOutput]
    decision: AbstentionDecision
    answer: str
    refinement_passes: int


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class EGAAgent:
    """The EGA reasoning agent.

    Parameters
    ----------
    tools : dict[str, BaseTool-like]
        Same shape MedRAX uses.
    llm : Any
        A langchain BaseLanguageModel. Used for claim decomposition and (optionally)
        for the LLM answer head. ``None`` falls back to rule-based decomposition.
    verifier, decomposer, answer_head, normalizer, collector, abstention
        Optional explicit overrides; sensible defaults are constructed if omitted.
    max_refinement_passes
        How many adaptive evidence-acquisition passes to run before the
        agent must commit to either an answer or an abstention.
    log_dir
        Where to write JSON logs of (question, graph, verifier outputs,
        decision). Set to None to disable logging.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        llm: Any = None,
        *,
        decomposer: Optional[BaseClaimDecomposer] = None,
        normalizer: Optional[EvidenceNormalizer] = None,
        collector: Optional[EvidenceCollector] = None,
        verifier: Optional[BaseVerifier] = None,
        abstention: Optional[AbstentionPolicy] = None,
        answer_head: Optional[AnswerHead] = None,
        reliability: Optional[ToolReliabilityPrior] = None,
        graph_builder: Optional[GraphBuilder] = None,
        max_refinement_passes: int = 2,
        log_dir: Optional[str] = "ega_logs",
    ):
        self.tools = tools
        self.llm = llm

        self.reliability = reliability or ToolReliabilityPrior()
        self.normalizer = normalizer or EvidenceNormalizer(reliability=self.reliability)
        self.collector = collector or EvidenceCollector(tools=tools, normalizer=self.normalizer)
        self.graph_builder = graph_builder or GraphBuilder()
        self.decomposer = decomposer or (
            LLMClaimDecomposer(llm) if llm is not None else RuleClaimDecomposer()
        )
        self.verifier = verifier or HybridVerifier(rule_verifier=RuleVerifier())
        self.abstention = abstention or AbstentionPolicy()
        self.answer_head = answer_head or TemplateAnswerHead()

        self.max_refinement_passes = max_refinement_passes
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.workflow = self._build_workflow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        image_path: str,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agent end-to-end.

        Returns a dict with: claims, graph (dict), verifier_outputs,
        decision, answer.
        """
        initial: EGAState = {
            "messages": [],
            "question": question,
            "image_path": image_path,
            "extra_inputs": extra_inputs or {},
            "refinement_passes": 0,
        }

        if self.workflow is not None:
            final_state = self.workflow.invoke(initial)
        else:
            # langgraph unavailable - run nodes synchronously in order.
            final_state = self._run_sequential(initial)

        result = self._serialize(final_state)
        if self.log_dir is not None:
            self._save_log(result)
        return result

    # ------------------------------------------------------------------
    # LangGraph wiring
    # ------------------------------------------------------------------

    def _build_workflow(self):
        if StateGraph is None:
            logger.warning("langgraph not installed (%s); using sequential runner", _LANGGRAPH_ERR)
            return None
        wf = StateGraph(EGAState)
        wf.add_node("decompose_claims", self._node_decompose)
        wf.add_node("collect_evidence", self._node_collect)
        wf.add_node("build_graph", self._node_build_graph)
        wf.add_node("verify_graph", self._node_verify)
        wf.add_node("refine", self._node_refine)
        wf.add_node("answer", self._node_answer)
        wf.set_entry_point("decompose_claims")
        wf.add_edge("decompose_claims", "collect_evidence")
        wf.add_edge("collect_evidence", "build_graph")
        wf.add_edge("build_graph", "verify_graph")
        wf.add_conditional_edges(
            "verify_graph",
            self._need_refinement,
            {True: "refine", False: "answer"},
        )
        wf.add_edge("refine", "build_graph")
        wf.add_edge("answer", END)
        return wf.compile()

    def _run_sequential(self, state: EGAState) -> EGAState:
        state.update(self._node_decompose(state))
        state.update(self._node_collect(state))
        state.update(self._node_build_graph(state))
        state.update(self._node_verify(state))
        while self._need_refinement(state):
            state.update(self._node_refine(state))
            state.update(self._node_build_graph(state))
            state.update(self._node_verify(state))
        state.update(self._node_answer(state))
        return state

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _node_decompose(self, state: EGAState) -> Dict[str, Any]:
        claims = self.decomposer(state["question"], state.get("extra_inputs", {}))
        return {"claims": claims, "graph": EvidenceGraph()}

    def _node_collect(self, state: EGAState) -> Dict[str, Any]:
        graph = state.get("graph") or EvidenceGraph()
        if not graph.claims:
            self.collector.collect_initial(
                state["claims"], state["image_path"], graph,
                extra_inputs=state.get("extra_inputs"),
            )
        return {"graph": graph}

    def _node_build_graph(self, state: EGAState) -> Dict[str, Any]:
        graph, violations = self.graph_builder.finalize(state["graph"])
        return {"graph": graph, "violations": violations}

    def _node_verify(self, state: EGAState) -> Dict[str, Any]:
        outputs = self.verifier.verify(state["graph"], state.get("violations", []))
        return {"verifier_outputs": outputs}

    def _node_refine(self, state: EGAState) -> Dict[str, Any]:
        self.collector.refine(
            state.get("verifier_outputs", {}),
            state["image_path"],
            state["graph"],
            extra_inputs=state.get("extra_inputs"),
        )
        return {"refinement_passes": state.get("refinement_passes", 0) + 1}

    def _node_answer(self, state: EGAState) -> Dict[str, Any]:
        decision = self.abstention.decide(
            state["graph"], state.get("verifier_outputs", {}),
            violations=state.get("violations", []),
            ood_score=float(state.get("extra_inputs", {}).get("ood_score", 0.0)),
            question=state["question"],
        )
        decision.answer = self.answer_head.generate(
            state["question"], state["graph"],
            state.get("verifier_outputs", {}), decision,
        )
        return {"decision": decision, "answer": decision.answer}

    # ------------------------------------------------------------------
    # Routing predicate
    # ------------------------------------------------------------------

    def _need_refinement(self, state: EGAState) -> bool:
        if state.get("refinement_passes", 0) >= self.max_refinement_passes:
            return False
        outputs = state.get("verifier_outputs", {})
        if not outputs:
            return False
        # Refine if any claim is gappy or conflicting.
        return any(
            v.sufficiency < self.collector.config.sufficiency_threshold
            or v.conflict > self.collector.config.conflict_threshold
            for v in outputs.values()
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize(self, state: EGAState) -> Dict[str, Any]:
        graph = state.get("graph") or EvidenceGraph()
        decision: Optional[AbstentionDecision] = state.get("decision")
        return {
            "question": state.get("question", ""),
            "image_path": state.get("image_path", ""),
            "claims": [c.to_dict() for c in state.get("claims", [])],
            "graph": graph.to_dict(),
            "violations": [asdict(v) for v in state.get("violations", [])],
            "verifier_outputs": {
                cid: {
                    "p_true": v.p_true, "p_false": v.p_false,
                    "uncertainty": v.uncertainty, "conflict": v.conflict,
                    "sufficiency": v.sufficiency, "rationale": v.rationale,
                }
                for cid, v in state.get("verifier_outputs", {}).items()
            },
            "decision": decision.to_dict() if decision else None,
            "answer": state.get("answer", ""),
            "refinement_passes": state.get("refinement_passes", 0),
        }

    def _save_log(self, result: Dict[str, Any]) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"ega_{ts}.json"
        try:
            path.write_text(json.dumps(result, indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to write log %s: %s", path, e)

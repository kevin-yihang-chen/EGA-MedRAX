"""Answer head.

Converts verified claims into a final natural-language answer. Two
implementations:

  * TemplateAnswerHead - deterministic, template-driven; used as fallback
    and for ablation when we want to remove the LLM from the answer path.
  * LLMAnswerHead - asks the LLM to verbalize the claim-level conclusions
    plus their evidence; the LLM is forbidden from adding new findings
    (we instruct it to use only the verified claim list).

We isolate this so the evidence reasoning stays the same regardless of
whether the answer is templated or LLM-verbalized; this matters for
evaluation - the claim-level metric (`p_true` accuracy) is independent of
the answer head, which is exactly what the paper wants to show.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import (
    AbstentionDecision,
    AbstentionReason,
    ClaimType,
    EvidenceGraph,
    Polarity,
    VerifierOutput,
)


ANSWER_PROMPT = """You are summarising a chest X-ray analysis.

You may ONLY use the verified claims below. Do not introduce any other finding.
Be concise (<= 80 words). Mention each claim's verdict and the supporting tool(s).
If asked to abstain, explain the reason in one sentence and do not give a verdict.

User question:
\"\"\"{question}\"\"\"

Verified claims (JSON):
{claims_json}

Abstention:
{abstention}

Write the answer:"""


class AnswerHead:
    """Base interface."""

    def generate(
        self,
        question: str,
        graph: EvidenceGraph,
        verifier_outputs: Dict[str, VerifierOutput],
        decision: AbstentionDecision,
    ) -> str:
        raise NotImplementedError


class TemplateAnswerHead(AnswerHead):
    """Deterministic verbalization - useful for ablation and tests."""

    def generate(
        self,
        question: str,
        graph: EvidenceGraph,
        verifier_outputs: Dict[str, VerifierOutput],
        decision: AbstentionDecision,
    ) -> str:
        if decision.abstain:
            return _abstain_text(decision)
        lines: List[str] = []
        for cid, v in verifier_outputs.items():
            claim = graph.claims.get(cid)
            if claim is None:
                continue
            verdict = "supported" if v.p_true >= v.p_false else "not supported"
            polarity_aware = self._verbalize(claim.text, claim.polarity, verdict)
            tools = ", ".join(sorted({e.tool_name for e in graph.evidence_for(cid)}))
            lines.append(f"- {polarity_aware} (confidence={v.confidence:.2f}; tools={tools})")
        return "Findings:\n" + "\n".join(lines)

    @staticmethod
    def _verbalize(text: str, polarity: Polarity, verdict: str) -> str:
        if verdict == "supported":
            return f"{text} - {'confirmed' if polarity != Polarity.NEGATIVE else 'ruled out'}"
        return f"{text} - {'not confirmed' if polarity != Polarity.NEGATIVE else 'cannot be ruled out'}"


class LLMAnswerHead(AnswerHead):
    """LLM-verbalized answer constrained to the verified claim list."""

    def __init__(self, llm: Any):
        self.llm = llm

    def generate(
        self,
        question: str,
        graph: EvidenceGraph,
        verifier_outputs: Dict[str, VerifierOutput],
        decision: AbstentionDecision,
    ) -> str:
        if decision.abstain:
            return _abstain_text(decision)
        claims_payload = []
        for cid, v in verifier_outputs.items():
            claim = graph.claims.get(cid)
            if claim is None:
                continue
            claims_payload.append({
                "claim": claim.text,
                "type": claim.type.value,
                "polarity": claim.polarity.value,
                "p_true": round(v.p_true, 3),
                "p_false": round(v.p_false, 3),
                "tools": sorted({e.tool_name for e in graph.evidence_for(cid)}),
            })
        prompt = ANSWER_PROMPT.format(
            question=question.strip(),
            claims_json=_safe_json(claims_payload),
            abstention="none",
        )
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", response))


def _abstain_text(decision: AbstentionDecision) -> str:
    reason_text = {
        AbstentionReason.INSUFFICIENT_VISUAL_EVIDENCE: "the available imaging evidence is insufficient",
        AbstentionReason.CROSS_TOOL_CONFLICT_UNRESOLVED: "the imaging tools disagree and the conflict could not be resolved",
        AbstentionReason.MISSING_REQUIRED_CONTEXT: "required context (e.g. a prior study) is missing",
        AbstentionReason.LOW_CONFIDENCE: "the verifier confidence is too low",
        AbstentionReason.OUT_OF_DISTRIBUTION: "the input appears to be out of distribution",
    }.get(decision.reason, "the agent could not reach a reliable conclusion")
    return (
        f"I am abstaining from a definitive answer because {reason_text}. "
        f"Abstention score={decision.abstention_score:.2f}, "
        f"max claim confidence={decision.confidence:.2f}."
    )


def _safe_json(payload: Any) -> str:
    import json
    return json.dumps(payload, indent=2, default=str)

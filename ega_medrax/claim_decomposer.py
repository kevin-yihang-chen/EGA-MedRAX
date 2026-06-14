"""Module A - Claim Decomposer.

Turns a natural-language clinical query into a set of atomic, judge-able
Claim objects. Implements two pathways:

  * LLMClaimDecomposer: prompts an LLM to emit JSON, then parses through a
    schema validator. Used at inference time.
  * RuleClaimDecomposer: a small offline fallback that uses keyword matching
    against the MedRAX pathology vocabulary. Used for unit tests, pseudo-
    labeling, and when no LLM is available.

The LLM pathway uses constrained decoding by re-asking on schema errors -
we keep it inside this module rather than wiring it into the agent loop so
the decomposer can be reused for offline label generation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .schemas import Claim, ClaimType, Polarity

# Pathologies recognised by the MedRAX classification tool.
PATHOLOGIES: Sequence[str] = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass",
    "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
)

_LATERALITY = {
    "left": "left", "right": "right", "bilateral": "bilateral",
    "both": "bilateral", "lt": "left", "rt": "right",
}


CLAIM_DECOMPOSITION_PROMPT = """You are a medical reasoning assistant.

Decompose the user's question about a chest X-ray into a list of ATOMIC
medical claims that can each be independently verified. Each claim must
isolate one (finding, location, severity, comparison, diagnosis, device,
or metadata) judgement.

Return a JSON array. Each element has the keys:
  - text       : human-readable statement
  - type       : one of {types}
  - subject    : the entity (e.g. "pleural effusion", "ET tube")
  - attribute  : qualifier (e.g. "present", "mild", "worsened")
  - anatomy    : "" or anatomical location (e.g. "left lower zone")
  - time       : "current" / "prior" / "delta"
  - polarity   : "positive" / "negative" / "uncertain"

Rules:
  * A finding-with-location-with-comparison question yields 3 claims.
  * "No effusion" is one claim with polarity=negative.
  * Diagnosis claims must depend on at least one finding claim.
  * Do not invent findings that the user did not ask about.

User question:
\"\"\"{question}\"\"\"

JSON only, no commentary:"""


class BaseClaimDecomposer:
    def __call__(self, question: str, context: Optional[Dict[str, Any]] = None) -> List[Claim]:
        return self.decompose(question, context or {})

    def decompose(self, question: str, context: Dict[str, Any]) -> List[Claim]:
        raise NotImplementedError


class LLMClaimDecomposer(BaseClaimDecomposer):
    """Prompt the LLM, parse JSON, validate against the Claim schema.

    Retries up to ``max_retries`` times on parse / schema failure with a
    tightened prompt; falls back to the rule decomposer after the budget
    is spent so the pipeline never hard-fails.
    """

    def __init__(self, llm: Any, max_retries: int = 2, fallback: Optional[BaseClaimDecomposer] = None):
        self.llm = llm
        self.max_retries = max_retries
        self.fallback = fallback or RuleClaimDecomposer()

    def decompose(self, question: str, context: Dict[str, Any]) -> List[Claim]:
        types_list = ", ".join(t.value for t in ClaimType)
        prompt = CLAIM_DECOMPOSITION_PROMPT.format(types=types_list, question=question.strip())

        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            if attempt == 0:
                raw = self._invoke(prompt)
            else:
                retry_prompt = (
                    prompt
                    + f"\n\nPrevious attempt failed with: {last_err}\n"
                    "Return ONLY a JSON array conforming exactly to the schema."
                )
                raw = self._invoke(retry_prompt)
            try:
                parsed = _extract_json_array(raw)
                claims = [_validate_claim_dict(c) for c in parsed]
                if claims:
                    _link_diagnosis_to_findings(claims)
                    return claims
                last_err = "empty array"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)

        return self.fallback.decompose(question, context)

    def _invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content)


class RuleClaimDecomposer(BaseClaimDecomposer):
    """Keyword-driven decomposer.

    Covers the common cases ("is there X on the left", "has X worsened
    compared with prior", bare finding questions). If it can't match a
    pattern it emits a single FINDING claim for the whole question so the
    downstream pipeline still has something to verify.
    """

    def decompose(self, question: str, context: Dict[str, Any]) -> List[Claim]:
        text = question.lower()
        claims: List[Claim] = []

        side = self._detect_side(text)
        subjects = self._detect_pathologies(text)
        polarity = Polarity.NEGATIVE if re.search(r"\b(no|without|absence of)\b", text) else Polarity.POSITIVE

        if not subjects:
            claims.append(Claim(
                text=question.strip(),
                type=ClaimType.FINDING,
                subject="unspecified",
                attribute="present",
                polarity=polarity,
            ))
            return claims

        finding_ids: List[str] = []
        for subject in subjects:
            finding = Claim(
                text=f"{subject} is {'absent' if polarity == Polarity.NEGATIVE else 'present'}"
                     + (f" on the {side}" if side else ""),
                type=ClaimType.FINDING,
                subject=subject.lower(),
                attribute="absent" if polarity == Polarity.NEGATIVE else "present",
                anatomy=f"{side} hemithorax" if side else "",
                polarity=polarity,
            )
            claims.append(finding)
            finding_ids.append(finding.id)

            if side:
                claims.append(Claim(
                    text=f"the {subject} is on the {side} side",
                    type=ClaimType.LOCATION,
                    subject=subject.lower(),
                    attribute=side,
                    anatomy=f"{side} hemithorax",
                    polarity=Polarity.POSITIVE,
                    parents=[finding.id],
                ))

        if re.search(r"\b(worsened|improved|changed|compared|prior|previous)\b", text):
            for fid, subject in zip(finding_ids, subjects):
                claims.append(Claim(
                    text=f"the {subject} has changed compared with the prior study",
                    type=ClaimType.COMPARISON,
                    subject=subject.lower(),
                    attribute="changed",
                    time="delta",
                    polarity=Polarity.POSITIVE,
                    parents=[fid],
                ))

        return claims

    @staticmethod
    def _detect_side(text: str) -> str:
        for key, side in _LATERALITY.items():
            if re.search(rf"\b{key}\b", text):
                return side
        return ""

    @staticmethod
    def _detect_pathologies(text: str) -> List[str]:
        found: List[str] = []
        for path in PATHOLOGIES:
            if re.search(rf"\b{re.escape(path.lower())}\b", text):
                found.append(path)
        synonyms = {
            "pleural effusion": "Effusion",
            "ptx": "Pneumothorax",
            "consolidation": "Consolidation",
            "fluid": "Effusion",
        }
        for k, v in synonyms.items():
            if k in text and v not in found:
                found.append(v)
        return found


def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON array in response")
    return json.loads(match.group(0))


def _validate_claim_dict(d: Dict[str, Any]) -> Claim:
    if "text" not in d:
        raise ValueError("claim missing 'text'")
    try:
        ctype = ClaimType(d.get("type", "finding"))
    except ValueError:
        ctype = ClaimType.FINDING
    try:
        pol = Polarity(d.get("polarity", "positive"))
    except ValueError:
        pol = Polarity.POSITIVE
    return Claim(
        text=str(d["text"]).strip(),
        type=ctype,
        subject=str(d.get("subject", "")).strip(),
        attribute=str(d.get("attribute", "")).strip(),
        anatomy=str(d.get("anatomy", "")).strip(),
        time=str(d.get("time", "current")).strip() or "current",
        polarity=pol,
    )


def _link_diagnosis_to_findings(claims: List[Claim]) -> None:
    """Attach DIAGNOSIS claims to all preceding FINDING claims as parents.

    The verifier consults these parents and refuses high-confidence
    diagnosis output unless at least one supporting finding has been
    verified.
    """
    finding_ids = [c.id for c in claims if c.type == ClaimType.FINDING]
    for c in claims:
        if c.type == ClaimType.DIAGNOSIS and not c.parents:
            c.parents = list(finding_ids)

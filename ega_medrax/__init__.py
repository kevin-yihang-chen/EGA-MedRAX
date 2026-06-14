"""EGA-MedRAX: Evidence-Graph Agent for Reliable Chest X-ray Reasoning.

A claim-centric medical reasoning agent built on top of MedRAX. The agent
decomposes a query into atomic medical claims, collects heterogeneous tool
evidence, structures everything as an evidence graph, runs a graph verifier
with explicit support/contradiction reasoning, and either answers with
calibrated confidence or abstains with a structured reason.
"""

from .schemas import (
    Claim,
    ClaimType,
    Polarity,
    Region,
    Evidence,
    EvidenceType,
    Proposition,
    EvidenceGraph,
    VerifierOutput,
    AbstentionDecision,
    AbstentionReason,
)
from .agent import EGAAgent, EGAState

__all__ = [
    "Claim",
    "ClaimType",
    "Polarity",
    "Region",
    "Evidence",
    "EvidenceType",
    "Proposition",
    "EvidenceGraph",
    "VerifierOutput",
    "AbstentionDecision",
    "AbstentionReason",
    "EGAAgent",
    "EGAState",
]

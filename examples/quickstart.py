"""Offline EGA-MedRAX demo.

Runs three diagnostic cases through the agent using mock tools:

  1. left effusion - tools agree, agent answers with high confidence.
  2. conflicting   - classifier vs VQA disagree, agent should abstain
                     with CROSS_TOOL_CONFLICT_UNRESOLVED.
  3. comparison_no_prior - prior study is required but absent, agent
                           should abstain with MISSING_REQUIRED_CONTEXT.

Run from the EGA-MedRAX root:

    PYTHONPATH=. python examples/quickstart.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ega_medrax import EGAAgent  # noqa: E402
from examples.mock_tools import build_mock_tools  # noqa: E402


CASES = [
    ("left_effusion",
     "Is there a left pleural effusion?",
     "demo/left_effusion.png"),
    ("conflicting",
     "Is there a pneumothorax?",
     "demo/conflict.png"),
    ("comparison_no_prior",
     "Has the pleural effusion worsened compared with the prior study?",
     "demo/no_prior.png"),
]


def main() -> None:
    for case, question, image in CASES:
        print("=" * 72)
        print(f"Case: {case}")
        print(f"Q: {question}")
        tools = build_mock_tools(case)
        agent = EGAAgent(tools=tools, log_dir=None, max_refinement_passes=1)
        result = agent.run(question, image_path=image)
        print(f"Answer: {result['answer']}")
        print(f"Decision: {result['decision']['reason']} "
              f"(abstain={result['decision']['abstain']}, "
              f"score={result['decision']['abstention_score']:.3f})")
        print("Verifier:")
        for cid, v in result["verifier_outputs"].items():
            claim = result["graph"]["claims"][cid]["text"]
            print(f"  - {claim[:60]:60s}  p_true={v['p_true']:.2f}  conflict={v['conflict']:.2f}")
        print()


if __name__ == "__main__":
    main()

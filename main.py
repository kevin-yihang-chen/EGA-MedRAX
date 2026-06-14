"""Entrypoint that mirrors MedRAX's ``main.py`` but boots the EGA agent.

Differences vs the upstream MedRAX entrypoint:

  * builds an ``EGAAgent`` instead of MedRAX's ``Agent``;
  * loads the trained verifier checkpoint if present;
  * keeps the same MedRAX-style tool registry so existing tool selections
    and model weight directories work unchanged.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
load_dotenv()


def _load_medrax_tools(
    selected_tools: List[str],
    model_dir: str,
    temp_dir: str,
    device: str,
) -> Dict[str, Any]:
    """Reuse the MedRAX tool factories.

    Requires the upstream MedRAX package to be importable. Add the cloned
    MedRAX repo to ``PYTHONPATH`` before launching this script.
    """
    from medrax.tools import (
        ChestXRayClassifierTool, ChestXRaySegmentationTool, ChestXRayReportGeneratorTool,
        ChestXRayGeneratorTool, XRayVQATool, LlavaMedTool, XRayPhraseGroundingTool,
        ImageVisualizerTool, DicomProcessorTool,
    )

    factories: Dict[str, Any] = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(cache_dir=model_dir, device=device),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device,
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device,
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
    }

    # The MedRAX BaseTools advertise themselves under a ``name`` attribute
    # that matches our tool registry keys, so we re-key by that name.
    out: Dict[str, Any] = {}
    for cls_name in selected_tools:
        if cls_name not in factories:
            continue
        tool = factories[cls_name]()
        out[getattr(tool, "name", cls_name)] = tool
    return out


def build_agent(
    *,
    selected_tools: List[str],
    model_dir: str,
    temp_dir: str,
    device: str,
    llm_model: str,
    temperature: float,
    top_p: float,
    verifier_ckpt: Optional[str],
    verifier_kind: str,
    log_dir: str,
) -> Any:
    from langchain_openai import ChatOpenAI

    from ega_medrax import EGAAgent
    from ega_medrax.verifier import (
        HybridVerifier,
        RuleVerifier,
        MLPClaimVerifier,
        GraphTransformerVerifier,
    )

    openai_kwargs: Dict[str, Any] = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    llm = ChatOpenAI(model=llm_model, temperature=temperature, top_p=top_p, **openai_kwargs)
    tools = _load_medrax_tools(selected_tools, model_dir, temp_dir, device)

    learned = None
    if verifier_ckpt:
        if verifier_kind == "graph":
            learned = GraphTransformerVerifier.load(verifier_ckpt, device=device)
        else:
            learned = MLPClaimVerifier.load(verifier_ckpt, device=device)

    verifier = HybridVerifier(rule_verifier=RuleVerifier(), learned_verifier=learned)

    agent = EGAAgent(
        tools=tools,
        llm=llm,
        verifier=verifier,
        log_dir=log_dir,
        max_refinement_passes=2,
    )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_dir", default="/model-weights")
    parser.add_argument("--temp_dir", default="temp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm_model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--verifier_ckpt", default=None)
    parser.add_argument("--verifier_kind", choices=["mlp", "graph"], default="mlp")
    parser.add_argument("--log_dir", default="ega_logs")
    parser.add_argument(
        "--tools", nargs="+",
        default=[
            "ImageVisualizerTool", "DicomProcessorTool",
            "ChestXRayClassifierTool", "ChestXRaySegmentationTool",
            "ChestXRayReportGeneratorTool", "XRayVQATool",
        ],
    )
    args = parser.parse_args()

    agent = build_agent(
        selected_tools=args.tools,
        model_dir=args.model_dir,
        temp_dir=args.temp_dir,
        device=args.device,
        llm_model=args.llm_model,
        temperature=args.temperature,
        top_p=args.top_p,
        verifier_ckpt=args.verifier_ckpt,
        verifier_kind=args.verifier_kind,
        log_dir=args.log_dir,
    )

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    result = agent.run(args.question, image_path=args.image)
    print(result["answer"])
    print("decision:", result["decision"])


if __name__ == "__main__":
    main()

# EGA-MedRAX

**Evidence-Graph Agent for Reliable Chest X-ray Reasoning**

EGA-MedRAX is an improvement over [MedRAX](https://github.com/bowang-lab/MedRAX)
that replaces MedRAX's two-node `process -> execute` ReAct loop with a
**six-node, claim-centric evidence-reasoning pipeline**. The agent:

1. **Decomposes** the user's query into atomic medical claims.
2. **Collects** heterogeneous evidence from chest X-ray tools (classifier,
   segmentation, grounding, VQA, report generator, DICOM metadata).
3. **Normalizes** all tool outputs into a single typed `Evidence` schema
   with explicit `SUPPORTS / CONTRADICTS / INSUFFICIENT` propositions.
4. **Builds an evidence graph** over claim / evidence / region nodes with
   four typed edges (`supports`, `contradicts`, `grounds`, `relates`).
5. **Verifies** each claim with a hybrid rule + graph-transformer verifier
   that produces per-claim `p_true`, `conflict`, and `sufficiency`.
6. **Answers or abstains** through a structured abstention policy with
   typed reasons (`INSUFFICIENT_VISUAL_EVIDENCE`,
   `CROSS_TOOL_CONFLICT_UNRESOLVED`, `MISSING_REQUIRED_CONTEXT`, ...).

The agent uses the same MedRAX `BaseTool` interface, so existing tool
selections and weights work without modification.

## Repository layout

```
EGA-MedRAX/
├── ega_medrax/
│   ├── schemas.py              # Claim / Evidence / Region / EvidenceGraph
│   ├── claim_decomposer.py     # Module A
│   ├── evidence/
│   │   ├── normalizer.py       # Module C (the keystone)
│   │   ├── tool_reliability.py # r(tool, claim_type)
│   │   └── collector.py        # Module B + adaptive refinement
│   ├── graph/
│   │   ├── builder.py          # facade + torch_geometric exporter
│   │   └── rules.py            # medical consistency rules
│   ├── verifier/
│   │   ├── rule_verifier.py
│   │   ├── learned_verifier.py # MLP + Hetero graph transformer
│   │   └── base.py             # HybridVerifier
│   ├── abstention/
│   │   ├── policy.py
│   │   └── answer_head.py
│   ├── agent.py                # LangGraph workflow
│   └── prompts/
├── train/
│   ├── losses.py               # claim + conflict + calib + abstention
│   ├── pseudo_label.py         # Stage 1 weak supervision
│   ├── dataset.py
│   └── train_verifier.py       # Stage 2 trainer
├── examples/
│   ├── mock_tools.py
│   └── quickstart.py
├── main.py                     # production entrypoint
└── requirements.txt
```

## Quickstart (no model weights required)

```bash
cd EGA-MedRAX
pip install -r requirements.txt    # torch + langgraph; torch-geometric optional
PYTHONPATH=. python examples/quickstart.py
```

The quickstart runs three diagnostic cases against mock tools and prints
the per-claim verifier output, the structured abstention decision, and
the final answer.

## Real MedRAX tools

To run the agent on a real chest X-ray with the MedRAX tool stack:

```bash
git clone https://github.com/bowang-lab/MedRAX
export PYTHONPATH=$PWD/MedRAX:$PWD/EGA-MedRAX

python EGA-MedRAX/main.py \
    --question "Is there a left pleural effusion?" \
    --image /path/to/cxr.png \
    --model_dir /model-weights \
    --tools ChestXRayClassifierTool ChestXRaySegmentationTool \
            XRayPhraseGroundingTool XRayVQATool
```

`main.py` accepts an optional `--verifier_ckpt` so you can plug a trained
MLP or graph-transformer verifier in.

## Training the verifier

Two-stage training, following the design document:

```bash
# Stage 1: pseudo-label a dataset of (question, image, reference answer)
python -m train.pseudo_label --in benchmark/samples.jsonl --out data/ega_pseudo

# Stage 2: fit the learned verifier
python -m train.train_verifier --data data/ega_pseudo \
    --model graph --output checkpoints/ega_v2.pt
```

The combined loss `L = L_claim + L_conflict + L_calib + L_abstain`
implements the four-term objective from the design doc, including the
Geifman & El-Yaniv selective-prediction term that wires the abstention
head into training.

## Design choices

The design is documented at length in `../MedRAX/Improvement.md`. Key
decisions that show up in the code:

- **Claim-centric, not response-centric.** Module A produces a `Claim`
  set; everything downstream operates per claim. The LLM is **not** the
  evidence fusion engine - the verifier is.
- **Evidence fusion, not message accumulation.** Tool outputs are
  normalized into typed `Evidence` nodes with explicit propositions, so
  conflict is detected as data, not "noticed" in free text.
- **Abstention is a first-class output.** The agent can refuse with a
  typed reason; the abstention policy is loss-trained, not threshold-
  tuned alone.

## Citation

If you use this code, please cite the original MedRAX paper:

```
@inproceedings{medrax2025,
  title={MedRAX: Medical Reasoning Agent for Chest X-ray},
  booktitle={ICML},
  year={2025}
}
```

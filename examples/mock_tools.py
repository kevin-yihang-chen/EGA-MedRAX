"""Mock chest X-ray tools used to exercise the EGA pipeline end-to-end
without needing the heavy MedRAX model weights. The mock outputs are
shaped exactly like the real tools so the normalizer can ingest them
unmodified - which is precisely the contract this method paper is making.
"""

from __future__ import annotations

from typing import Any, Dict


class MockClassifier:
    name = "chest_xray_classifier"

    def __init__(self, scores: Dict[str, float]):
        self.scores = scores

    def invoke(self, args: Dict[str, Any]):
        return self.scores, {"image_path": args.get("image_path"), "analysis_status": "completed"}


class MockSegmentation:
    name = "chest_xray_segmentation"

    def __init__(self, masks: Dict[str, Dict[str, Any]]):
        self.masks = masks

    def invoke(self, args: Dict[str, Any]):
        return {"masks": self.masks}, {"image_path": args.get("image_path")}


class MockGrounding:
    name = "xray_phrase_grounding"

    def __init__(self, boxes: Dict[str, Dict[str, Any]]):
        self.boxes = boxes

    def invoke(self, args: Dict[str, Any]):
        phrase = args.get("phrase", "")
        return {"boxes": {k: v for k, v in self.boxes.items() if phrase.lower() in k.lower()}}, {}


class MockVQA:
    name = "xray_vqa"

    def __init__(self, response_map: Dict[str, str]):
        self.response_map = response_map

    def invoke(self, args: Dict[str, Any]):
        q = args.get("question", "").lower()
        for key, ans in self.response_map.items():
            if key in q:
                return {"answer": ans}, {}
        return {"answer": "unable to determine"}, {}


def build_mock_tools(case: str = "left_effusion") -> Dict[str, Any]:
    """Curated cases used by the quickstart demo."""
    if case == "left_effusion":
        return {
            "chest_xray_classifier": MockClassifier({
                "Effusion": 0.82, "Cardiomegaly": 0.15, "Pneumothorax": 0.05,
                "Consolidation": 0.20, "Pneumonia": 0.12,
            }),
            "chest_xray_segmentation": MockSegmentation({
                "effusion-left-base": {"area": 0.12, "side": "left",
                                        "zone": "lower", "bbox": [0.05, 0.5, 0.4, 0.95]},
            }),
            "xray_phrase_grounding": MockGrounding({
                "left pleural effusion": {"score": 0.74, "bbox": [0.05, 0.55, 0.4, 0.92]},
            }),
            "xray_vqa": MockVQA({"effusion": "Yes, a small left pleural effusion is present."}),
        }
    if case == "conflicting":
        return {
            "chest_xray_classifier": MockClassifier({"Pneumothorax": 0.61, "Effusion": 0.20}),
            "chest_xray_segmentation": MockSegmentation({
                "pneumothorax": {"area": 0.001, "side": "right", "bbox": [0.6, 0.1, 0.8, 0.4]},
            }),
            "xray_vqa": MockVQA({"pneumothorax": "No definite pneumothorax is seen."}),
        }
    if case == "comparison_no_prior":
        return {
            "chest_xray_classifier": MockClassifier({"Effusion": 0.65}),
            "xray_vqa": MockVQA({"effusion": "There is an effusion in the current image."}),
        }
    raise ValueError(f"unknown mock case: {case}")

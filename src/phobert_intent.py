import os
import sys
import json
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INTENT_LABELS, PHOBERT_MODEL_PATH


class PhoBERTIntentClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or PHOBERT_MODEL_PATH
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._labels = None

    def is_available(self) -> bool:
        return os.path.exists(self.model_path) and os.path.isdir(self.model_path)

    def _load_labels(self):
        label_path = os.path.join(self.model_path, "label_config.json")
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as f:
                cfg = json.load(f)
            id2label = cfg.get("id2label", {})
            self._labels = [id2label[str(i)] for i in range(len(id2label))]
        else:
            self._labels = INTENT_LABELS

    def load(self):
        if not self.is_available():
            raise FileNotFoundError(
                f"PhoBERT model không tìm thấy tại {self.model_path}\n"
                f"Train trên Colab (train_phobert_intent_colab.ipynb) và copy folder về."
            )

        print(f"[PhoBERT] Loading model từ {self.model_path}...")
        self._load_labels()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[PhoBERT] Model loaded (device: {self.device})")

    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        if self.model is None:
            self.load()

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values

        pred_labels = [self._labels[p.item()] for p in predictions]
        pred_confs = confidences.cpu().numpy().tolist()
        return pred_labels, pred_confs

    def predict_single(self, text: str) -> dict:
        labels, confs = self.predict([text])
        return {
            "intent": labels[0],
            "confidence": round(confs[0], 4),
            "source": "phobert",
        }

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        from sklearn.metrics import classification_report, accuracy_score, f1_score

        pred_labels, pred_confs = self.predict(texts)
        report = classification_report(
            labels,
            pred_labels,
            target_names=self._labels,
            zero_division=0,
        )
        return {
            "predictions": pred_labels,
            "confidences": pred_confs,
            "accuracy": accuracy_score(labels, pred_labels),
            "f1_macro": f1_score(labels, pred_labels, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                labels, pred_labels, average="weighted", zero_division=0
            ),
            "report": report,
        }


def check_phobert_availability() -> bool:
    return PhoBERTIntentClassifier().is_available()

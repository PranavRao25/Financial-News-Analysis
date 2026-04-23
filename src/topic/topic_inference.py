import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parent = Path(__file__).resolve().parent.parent.parent

class TopicModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initializes the inference engine by loading the saved model and tokenizer.
        """

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Processes a batch of raw strings and returns class predictions with confidence scores.
        """

        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        confidences, predicted_classes = torch.max(probabilities, dim=-1)
        
        results = []
        for text, pred_class, conf in zip(texts, predicted_classes, confidences):
            results.append({
                "text": text,
                "predicted_class": pred_class.item(),
                "confidence": conf.item()
            })
        return results

if __name__ == "__main__":
    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)
    model_path = configs["topic"]["model"]["path"]
    classifier = TopicModel(model_path, device="cpu")
    
    sample_texts = [
        "The Federal Reserve announced a 50 basis point interest rate hike today, impacting bond yields.",
        "The tech giant's Q3 earnings report showed a 15% revenue miss, sending shares tumbling in after-hours trading."
    ]
    predictions = classifier.predict(sample_texts)
    
    print("\n--- Inference Results ---")
    for pred in predictions:
        print(f"Input: {pred['text']}")
        print(f"Class ID: {pred['predicted_class']} | Confidence: {pred['confidence']:.4f}\n")

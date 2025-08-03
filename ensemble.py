# ensemble.py

from detector import DeepfakeDetector

class EnsembleDeepfakeDetector:
    def __init__(self):
        self.models = {}
        self.weights = {}

    def add_model(self, name, model, weight=1.0):
        self.models[name] = model
        self.weights[name] = weight

    def predict_ensemble(self, video_path):
        predictions = {}

        for name, model in self.models.items():
            pred, _ = model.predict_video(video_path)
            predictions[name] = pred

        weighted_sum = sum(pred * self.weights[name] for name, pred in predictions.items())
        total_weight = sum(self.weights.values())

        ensemble_prediction = weighted_sum / total_weight
        return ensemble_prediction, predictions

# Example usage (if needed)
if __name__ == "__main__":
    detector1 = DeepfakeDetector()
    detector2 = DeepfakeDetector()
    detector3 = DeepfakeDetector()
    
    ensemble = EnsembleDeepfakeDetector()
    ensemble.add_model("xception_lstm", detector1, weight=0.4)
    ensemble.add_model("efficientnet_lstm", detector2, weight=0.3)
    ensemble.add_model("resnet_lstm", detector3, weight=0.3)

    prediction, individual_preds = ensemble.predict_ensemble("test_videos/test_video.mp4")
    print("Final Prediction:", prediction)
    print("Individual Predictions:", individual_preds)

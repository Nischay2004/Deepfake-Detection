# train.py

from detector import DeepfakeDetector
import numpy as np
from sklearn.model_selection import train_test_split

def train_deepfake_detector():
    detector = DeepfakeDetector()

    real_videos = ['dataset/real/real1.mp4', 'dataset/real/real2.mp4']
    fake_videos = ['dataset/fake/fake1.mp4', 'dataset/fake/fake2.mp4']

    X_train, y_train = [], []

    for video in real_videos:
        seqs, labels = detector.preprocess_video_for_training(video, 1)
        if seqs is not None:
            X_train.extend(seqs)
            y_train.extend(labels)

    for video in fake_videos:
        seqs, labels = detector.preprocess_video_for_training(video, 0)
        if seqs is not None:
            X_train.extend(seqs)
            y_train.extend(labels)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("Training CNN model...")
    detector.train_cnn_model(X_train, y_train, X_val, y_val)

    print("Training LSTM model...")
    detector.train_lstm_model(X_train, y_train, X_val, y_val)

    return detector

if __name__ == "__main__":
    train_deepfake_detector()

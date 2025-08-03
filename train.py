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
        seq, label = detector.preprocess_video_for_training(video, 1)
        if seq is not None:
            X_train.append(seq)
            y_train.append(label[0])  # Extract the scalar value from the array

    for video in fake_videos:
        seq, label = detector.preprocess_video_for_training(video, 0)
        if seq is not None:
            X_train.append(seq)
            y_train.append(label[0])  # Extract the scalar value from the array

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # For CNN training, we need to reshape the data to (batch*sequence_length, height, width, channels)
    # and repeat the labels for each frame in the sequence
    batch_size, seq_len, height, width, channels = X_train.shape
    X_train_cnn = X_train.reshape(-1, height, width, channels)
    y_train_cnn = np.repeat(y_train, seq_len)
    
    batch_size_val, seq_len_val, height_val, width_val, channels_val = X_val.shape
    X_val_cnn = X_val.reshape(-1, height_val, width_val, channels_val)
    y_val_cnn = np.repeat(y_val, seq_len_val)

    print(f"CNN training data shape: X_train_cnn={X_train_cnn.shape}, y_train_cnn={y_train_cnn.shape}")

    print("Training CNN model...")
    detector.train_cnn_model(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn)

    print("Training LSTM model...")
    detector.train_lstm_model(X_train, y_train, X_val, y_val)

    return detector

if __name__ == "__main__":
    train_deepfake_detector()

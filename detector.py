import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array

class DeepfakeDetector:
    def __init__(self, sequence_length=30, img_size=(160, 160)):
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.cnn_model = None
        self.lstm_model = None
        self.detector = MTCNN()

    def build_cnn_model(self):
        """Builds and compiles CNN (Xception) for frame-level feature extraction."""
        base = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        x = base.output
        x = GlobalAveragePooling2D(name='features')(x)
        out = Dense(1, activation='sigmoid', name='predictions')(x)
        self.cnn_model = Model(inputs=base.input, outputs=out)
        self.cnn_model.compile(
            optimizer=Adam(1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        # feature extractor for LSTM
        self.cnn_feature_extractor = Model(inputs=base.input, outputs=base.get_layer('features').output)

    def train_cnn_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=8):
        print("▶️ Training CNN...")
        if self.cnn_model is None:
            self.build_cnn_model()
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        print("✅ CNN training complete.")
        return history

    def build_lstm_model(self, feature_dim=None):
        """Builds and compiles LSTM for temporal sequence classification."""
        if feature_dim is None:
            raise ValueError("feature_dim must be specified")
        inp = Input(shape=(self.sequence_length, feature_dim))
        x = LSTM(128, return_sequences=False)(inp)
        x = Dropout(0.5)(x)
        out = Dense(1, activation='sigmoid')(x)
        self.lstm_model = Model(inputs=inp, outputs=out)
        self.lstm_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_lstm_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=4):
        print("▶️ Training LSTM...")
        feature_dim = X_train.shape[-1]
        self.build_lstm_model(feature_dim=feature_dim)
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        print("✅ LSTM training complete.")
        return history

    def extract_faces(self, frame):
        """Detects and crops the largest face in a single RGB frame."""
        detections = self.detector.detect_faces(frame)
        if not detections:
            return None
        # pick largest box
        box = max(detections, key=lambda d: d['box'][2] * d['box'][3])['box']
        x, y, w, h = box
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, self.img_size)
        return img_to_array(face)

    def preprocess_video(self, video_path, max_frames=None):
        """
        Reads video, detects up to `sequence_length` frames with faces,
        returns array shape (sequence_length, H, W, 3).
        """
        cap = cv2.VideoCapture(video_path)
        faces = []
        count = 0
        while cap.isOpened() and count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.extract_faces(rgb)
            if face is not None:
                faces.append(face)
                count += 1
        cap.release()
        if len(faces) < self.sequence_length:
            # pad with zeros
            pad_n = self.sequence_length - len(faces)
            h, w, c = self.img_size[0], self.img_size[1], 3
            faces += [np.zeros((h, w, c), dtype='float32')] * pad_n
        return np.array(faces, dtype='float32') / 255.0

    def preprocess_video_for_training(self, video_path, label):
        """
        Returns:
          X_seq: np.array of shape (sequence_length, H, W, 3)
          y: scalar label 0 or 1
        """
        seq = self.preprocess_video(video_path)
        return seq, np.array([label], dtype='float32')

    def extract_features_from_sequences(self, sequences):
        """
        sequences: list or array of shape (N, seq_len, H, W, 3)
        returns: np.array of shape (N, seq_len, feature_dim)
        """
        features = []
        for seq in sequences:
            # CNN feature extractor expects (batch, H, W, 3)
            feats = self.cnn_feature_extractor.predict(seq, verbose=0)
            features.append(feats)
        return np.array(features)

    def predict(self, video_path):
        """Full inference: video -> CNN features -> LSTM -> score."""
        seq = self.preprocess_video(video_path)
        feats = self.extract_features_from_sequences([seq])
        score = self.lstm_model.predict(feats, verbose=0)[0][0]
        return float(score)

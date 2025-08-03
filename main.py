# import tensorflow as tf
# from tensorflow.keras.applications import Xception
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTM, TimeDistributed
# from tensorflow.keras.models import Model, Sequential
# import cv2
# import numpy as np
# import pandas as pd
# from mtcnn import MTCNN
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# class DeepfakeDetector:
#     def __init__(self, sequence_length=20, image_size=(299, 299)):
#         self.sequence_length = sequence_length
#         self.image_size = image_size
#         self.face_detector = MTCNN()
#         self.cnn_model = None
#         self.lstm_model = None
#         self.ensemble_model = None
        
#     def extract_frames_from_video(self, video_path, max_frames=None):
#         """Extract frames from video file"""
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         frame_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             if max_frames and frame_count >= max_frames:
#                 break
                
#             frames.append(frame)
#             frame_count += 1
            
#         cap.release()
#         return frames
    
#     def detect_and_crop_faces(self, frames):
#         """Detect faces in frames and crop them"""
#         cropped_faces = []
        
#         for frame in frames:
#             # Convert BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Detect faces
#             faces = self.face_detector.detect_faces(rgb_frame)
            
#             if faces:
#                 # Get the largest face
#                 face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
#                 x, y, w, h = face['box']
                
#                 # Add margin
#                 margin = 20
#                 x1 = max(0, x - margin)
#                 y1 = max(0, y - margin)
#                 x2 = min(rgb_frame.shape[1], x + w + margin)
#                 y2 = min(rgb_frame.shape[0], y + h + margin)
                
#                 # Crop face
#                 cropped_face = rgb_frame[y1:y2, x1:x2]
                
#                 # Resize to target size
#                 cropped_face = cv2.resize(cropped_face, self.image_size)
#                 cropped_faces.append(cropped_face)
        
#         return np.array(cropped_faces)
    
#     def build_cnn_feature_extractor(self):
#         """Build CNN model for feature extraction"""
#         base_model = Xception(
#             weights='imagenet',
#             include_top=False,
#             input_shape=(*self.image_size, 3)
#         )
        
#         # Freeze base model layers
#         base_model.trainable = False
        
#         # Add custom layers
#         x = base_model.output
#         x = GlobalAveragePooling2D()(x)
#         x = Dense(512, activation='relu')(x)
#         features = Dense(256, activation='relu', name='features')(x)
        
#         # For training, add classification layer
#         predictions = Dense(1, activation='sigmoid', name='predictions')(features)
        
#         self.cnn_model = Model(inputs=base_model.input, outputs=[features, predictions])
        
#         return self.cnn_model
    
#     def build_lstm_model(self, cnn_feature_size=256):
#         """Build LSTM model for temporal analysis"""
#         model = Sequential([
#             TimeDistributed(Dense(128, activation='relu'), 
#                           input_shape=(self.sequence_length, cnn_feature_size)),
#             LSTM(64, return_sequences=True, dropout=0.5),
#             LSTM(32, dropout=0.5),
#             Dense(16, activation='relu'),
#             Dense(1, activation='sigmoid')
#         ])
        
#         self.lstm_model = model
#         return model
    
#     def preprocess_video_for_training(self, video_path, label):
#         """Preprocess a single video for training"""
#         frames = self.extract_frames_from_video(video_path)
        
#         if len(frames) < self.sequence_length:
#             return None, None
            
#         faces = self.detect_and_crop_faces(frames)
        
#         if len(faces) < self.sequence_length:
#             return None, None
        
#         # Create sequences
#         sequences = []
#         labels = []
        
#         for i in range(0, len(faces) - self.sequence_length + 1, self.sequence_length // 2):
#             sequence = faces[i:i + self.sequence_length]
#             if len(sequence) == self.sequence_length:
#                 sequences.append(sequence)
#                 labels.append(label)
        
#         return np.array(sequences), np.array(labels)
    
#     def train_cnn_model(self, X_train, y_train, X_val, y_val, epochs=10):
#         """Train CNN feature extractor"""
#         if self.cnn_model is None:
#             self.build_cnn_feature_extractor()
        
#         # Reshape data for CNN training
#         X_train_flat = X_train.reshape(-1, *self.image_size, 3)
#         y_train_flat = np.repeat(y_train, self.sequence_length)
        
#         X_val_flat = X_val.reshape(-1, *self.image_size, 3)
#         y_val_flat = np.repeat(y_val, self.sequence_length)
        
#         # Normalize
#         X_train_flat = X_train_flat.astype('float32') / 255.0
#         X_val_flat = X_val_flat.astype('float32') / 255.0
        
#         # Compile model
#         self.cnn_model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         # Train
#         history = self.cnn_model.fit(
#             X_train_flat, y_train_flat,
#             batch_size=32,
#             epochs=epochs,
#             validation_data=(X_val_flat, y_val_flat),
#             verbose=1
#         )
        
#         return history
    
#     def extract_features_from_sequences(self, sequences):
#         """Extract features from video sequences using trained CNN"""
#         if self.cnn_model is None:
#             raise ValueError("CNN model not trained yet")
        
#         # Get feature extractor (without classification layer)
#         feature_extractor = Model(
#             inputs=self.cnn_model.input,
#             outputs=self.cnn_model.get_layer('features').output
#         )
        
#         features = []
#         for sequence in sequences:
#             # Normalize
#             sequence_norm = sequence.astype('float32') / 255.0
            
#             # Extract features for each frame
#             sequence_features = feature_extractor.predict(sequence_norm, verbose=0)
#             features.append(sequence_features)
        
#         return np.array(features)
    
#     def train_lstm_model(self, X_train, y_train, X_val, y_val, epochs=20):
#         """Train LSTM model for temporal analysis"""
#         if self.lstm_model is None:
#             self.build_lstm_model()
        
#         # Extract features
#         X_train_features = self.extract_features_from_sequences(X_train)
#         X_val_features = self.extract_features_from_sequences(X_val)
        
#         # Compile model
#         self.lstm_model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         # Train
#         history = self.lstm_model.fit(
#             X_train_features, y_train,
#             batch_size=16,
#             epochs=epochs,
#             validation_data=(X_val_features, y_val),
#             verbose=1
#         )
        
#         return history
    
#     def predict_video(self, video_path):
#         """Predict if a video is deepfake"""
#         frames = self.extract_frames_from_video(video_path)
        
#         if len(frames) < self.sequence_length:
#             return 0.5, "Video too short for analysis"
        
#         faces = self.detect_and_crop_faces(frames)
        
#         if len(faces) < self.sequence_length:
#             return 0.5, "Not enough faces detected"
        
#         # Create sequences
#         sequences = []
#         for i in range(0, len(faces) - self.sequence_length + 1, self.sequence_length):
#             sequence = faces[i:i + self.sequence_length]
#             if len(sequence) == self.sequence_length:
#                 sequences.append(sequence)
        
#         if not sequences:
#             return 0.5, "No valid sequences found"
        
#         # Extract features and predict
#         sequences = np.array(sequences)
#         features = self.extract_features_from_sequences(sequences)
#         predictions = self.lstm_model.predict(features, verbose=0)
        
#         # Average predictions
#         avg_prediction = np.mean(predictions)
#         confidence = abs(avg_prediction - 0.5) * 2
        
#         result = "FAKE" if avg_prediction < 0.5 else "REAL"
        
#         return avg_prediction, f"{result} (Confidence: {confidence:.2f})"

# # Usage Example
# def train_deepfake_detector():
#     """Example training workflow"""
#     detector = DeepfakeDetector()
    
#     # Prepare training data
#     # You would load your dataset here
#     real_videos = ['path/to/real1.mp4', 'path/to/real2.mp4']
#     fake_videos = ['path/to/fake1.mp4', 'path/to/fake2.mp4']
    
#     X_train, y_train = [], []
    
#     # Process real videos
#     for video_path in real_videos:
#         sequences, labels = detector.preprocess_video_for_training(video_path, 1)
#         if sequences is not None:
#             X_train.extend(sequences)
#             y_train.extend(labels)
    
#     # Process fake videos
#     for video_path in fake_videos:
#         sequences, labels = detector.preprocess_video_for_training(video_path, 0)
#         if sequences is not None:
#             X_train.extend(sequences)
#             y_train.extend(labels)
    
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
    
#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42
#     )
    
#     # Train models
#     print("Training CNN model...")
#     cnn_history = detector.train_cnn_model(X_train, y_train, X_val, y_val)
    
#     print("Training LSTM model...")
#     lstm_history = detector.train_lstm_model(X_train, y_train, X_val, y_val)
    
#     return detector

# # Web Interface using Streamlit
# def create_streamlit_app():
#     """Create a Streamlit web interface"""
#     import streamlit as st
    
#     st.title("Deepfake Detection System")
#     st.write("Upload a video to check if it's a deepfake")
    
#     uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
#     if uploaded_file is not None:
#         # Save uploaded file
#         with open("temp_video.mp4", "wb") as f:
#             f.write(uploaded_file.read())
        
#         st.video("temp_video.mp4")
        
#         if st.button("Analyze Video"):
#             with st.spinner("Analyzing video..."):
#                 # Load trained model (you would load your saved model here)
#                 detector = DeepfakeDetector()
#                 # detector.load_models()  # Implement model loading
                
#                 prediction, result = detector.predict_video("temp_video.mp4")
                
#                 st.subheader("Results:")
#                 st.write(f"Prediction Score: {prediction:.3f}")
#                 st.write(f"Result: {result}")
                
#                 # Visualization
#                 if prediction < 0.5:
#                     st.error("⚠️ This video appears to be a DEEPFAKE!")
#                 else:
#                     st.success("✅ This video appears to be REAL!")
        
#         # Clean up
#         if os.path.exists("temp_video.mp4"):
#             os.remove("temp_video.mp4")

# if __name__ == "__main__":
#     # For training
#     # detector = train_deepfake_detector()
    
#     # For web app
#     create_streamlit_app()



# class EnsembleDeepfakeDetector:
#     def __init__(self):
#         self.models = {}
#         self.weights = {}
        
#     def add_model(self, name, model, weight=1.0):
#         """Add a model to the ensemble"""
#         self.models[name] = model
#         self.weights[name] = weight
    
#     def predict_ensemble(self, video_path):
#         """Make ensemble prediction"""
#         predictions = {}
        
#         for name, model in self.models.items():
#             pred, _ = model.predict_video(video_path)
#             predictions[name] = pred
        
#         # Weighted average
#         weighted_sum = sum(pred * self.weights[name] for name, pred in predictions.items())
#         total_weight = sum(self.weights.values())
        
#         ensemble_prediction = weighted_sum / total_weight
        
#         return ensemble_prediction, predictions

# # Ensemble usage
# ensemble = EnsembleDeepfakeDetector()
# ensemble.add_model("xception_lstm", detector1, weight=0.4)
# ensemble.add_model("efficientnet_lstm", detector2, weight=0.3)
# ensemble.add_model("resnet_lstm", detector3, weight=0.3)

# prediction, individual_preds = ensemble.predict_ensemble("test_video.mp4")


# def evaluate_model(detector, test_videos, test_labels):
#     """Comprehensive model evaluation"""
#     predictions = []
#     true_labels = []
    
#     for video_path, label in zip(test_videos, test_labels):
#         pred, _ = detector.predict_video(video_path)
#         predictions.append(1 if pred > 0.5 else 0)
#         true_labels.append(label)
    
#     # Calculate metrics
#     accuracy = accuracy_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions)
#     recall = recall_score(true_labels, predictions)
#     f1 = f1_score(true_labels, predictions)
    
#     print(f"Accuracy: {accuracy:.3f}")
#     print(f"Precision: {precision:.3f}")
#     print(f"Recall: {recall:.3f}")
#     print(f"F1-Score: {f1:.3f}")
    
#     # Confusion Matrix
#     cm = confusion_matrix(true_labels, predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
    
#     return accuracy, precision, recall, f1


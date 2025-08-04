# 🎭 Deepfake Detection System

A video-based deepfake detection system that combines **face detection**, **CNN-based feature extraction**, **temporal modeling**, and **ensemble learning**. This project also features a lightweight **Streamlit app** for quick testing of the model on new video data.

![Architecture](https://github.com/user-attachments/assets/ce4a2a76-6e95-4b67-98c0-2da5714f6aa4)

---

## 🧠 System Overview

This deepfake detection pipeline involves the following steps:

1. **🎞️ Frame Extraction**: Extract video frames from input videos.
2. **🧍‍♂️ Face Detection**: Use MTCNN or MediaPipe to detect and crop faces.
3. **🔍 Feature Extraction**: Use CNN architectures (e.g., Xception) for spatial analysis.
4. **⏳ Temporal Modeling**: Use RNNs or LSTMs to detect motion inconsistencies across frames.
5. **✅ Classification**: Real or Fake prediction.
6. **📈 Ensemble Learning**: Combine outputs from multiple models for more robust results.

---

## 📁 Project Structure

```bash
Deepfake_model/
│
├── dataset/              # Contains real and fake samples for training
│   ├── real/
│   └── fake/
│
├── test_videos/          # Videos to test the model against
│
├── detector.py           # Handles face detection and preprocessing
├── ensemble.py           # Combines multiple model outputs
├── train.py              # Training script for CNN + LSTM models
├── streamlit_app.py      # Web interface for uploading and testing videos
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
└── __pycache__/          # Compiled Python files
````

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Deepfake_model.git
cd Deepfake_model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 🧠 To Train the Model

Make sure the `dataset/real` and `dataset/fake` folders contain your training videos.

```bash
python train.py
```

### 🎬 To Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

This will launch a browser interface to upload test videos and get real/fake predictions.

---

## 📌 Key Components

| File               | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `train.py`         | Trains CNN and LSTM models using preprocessed face frames          |
| `detector.py`      | Uses MTCNN or MediaPipe to detect and crop faces from video frames |
| `ensemble.py`      | Combines predictions from multiple models using voting/averaging   |
| `streamlit_app.py` | Streamlit frontend for uploading and testing videos                |

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-Score
* ROC-AUC
* Frame-level and video-level inference

---

## 📦 Future Improvements

* Add real-time webcam-based detection
* Integrate transformer-based models (e.g., ViViT, TimeSformer)
* Model optimization for faster inference (e.g., TensorRT, ONNX export)

---

## 📬 Contact

For any inquiries or contributions, contact: **[your-email@example.com](mailto:your-email@example.com)**

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).



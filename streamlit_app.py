# streamlit_app.py

import streamlit as st
import os
from detector import DeepfakeDetector

def create_streamlit_app():
    st.title("Deepfake Detection System")
    st.write("Upload a video to check if it's a deepfake")

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        st.video("temp_video.mp4")

        if st.button("Analyze Video"):
            with st.spinner("Analyzing video..."):
                detector = DeepfakeDetector()
                # detector.load_models()  # Optional model loading
                prediction, result = detector.predict_video("temp_video.mp4")

                st.subheader("Results:")
                st.write(f"Prediction Score: {prediction:.3f}")
                st.write(f"Result: {result}")

                if prediction < 0.5:
                    st.error("⚠️ This video appears to be a DEEPFAKE!")
                else:
                    st.success("✅ This video appears to be REAL!")

        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

if __name__ == "__main__":
    create_streamlit_app()

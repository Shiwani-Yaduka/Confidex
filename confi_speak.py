import streamlit as st
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import sounddevice as sd
import librosa
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import random
import threading
import time

# Setup
st.set_page_config(page_title="Confidence Motivator App", layout="centered")
st.title("🎤 Confident Speaking Practice with Motivation Boost")

prompts = [
   
    "Give introduction of your self"
]

if st.button("🎲 Get Speaking Prompt"):
    st.info(random.choice(prompts))

if st.button("🎥 Start 15-sec Speaking Session"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 20)  # Try to increase FPS

    face_detector = FaceMeshDetector(maxFaces=1)
    hand_detector = HandDetector(maxHands=1, detectionCon=0.7)

    stframe = st.empty()
    progress_bar = st.progress(0)

    emotion_score = 0
    head_score = 0
    hand_activity_frames = 0
    prev_finger_count = 0

    total_frames = 0
    max_duration_sec = 15
    target_fps = 20
    max_frames = max_duration_sec * target_fps

    audio_data = []
    fs = 44100

    def record_audio():
        audio = sd.rec(int(max_duration_sec * fs), samplerate=fs, channels=1)
        sd.wait()
        audio_data.extend(audio)

    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

    start_time = time.time()
    while total_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, faces = face_detector.findFaceMesh(frame, draw=True)
        hands, frame = hand_detector.findHands(frame, draw=True)

        # Face / emotion detection: count smile based on lips distance
        if faces:
            face = faces[0]
            top_lip = face[13]
            bottom_lip = face[14]
            if abs(top_lip[1] - bottom_lip[1]) > 15:
                emotion_score += 1

            nose = face[1]
            # Count if head roughly center
            if 200 < nose[0] < 450:
                head_score += 1

        # Hand movement detection: count frames where finger count changes or fingers are up
        if hands:
            hand = hands[0]
            fingers = hand_detector.fingersUp(hand)
            finger_count = sum(fingers)
            # Count frame as "active" if fingers are moving or more than 1 finger up
            if finger_count != prev_finger_count or finger_count > 1:
                hand_activity_frames += 1
            prev_finger_count = finger_count

        total_frames += 1
        elapsed_time = time.time() - start_time
        progress_bar.progress(min(int((elapsed_time / max_duration_sec) * 100), 100))
        stframe.image(frame, channels="BGR")

    cap.release()
    audio_thread.join()
    st.success("✅ Visual and audio analysis complete.")

    # Audio analysis - better pitch extraction with librosa.yin
    audio_np = np.array(audio_data).flatten()
    # librosa requires float32 audio between -1 and 1
    audio_float = audio_np.astype(np.float32)
    audio_float /= np.max(np.abs(audio_float)) + 1e-9  # normalize to avoid div0

    try:
        f0 = librosa.yin(audio_float, fmin=50, fmax=500, sr=fs)
        f0 = f0[f0 > 0]  # remove zeros (unvoiced frames)
        pitch_mean = np.mean(f0)
        pitch_std = np.std(f0)
    except Exception as e:
        pitch_mean, pitch_std = 0, 0
        st.warning(f"Pitch extraction error: {e}")

    # Scoring with weighted factors
    emotion_ratio = emotion_score / total_frames
    head_ratio = head_score / total_frames
    hand_ratio = hand_activity_frames / total_frames

    # Weights: voice pitch 40%, hand 35%, emotion+head 25%
    voice_score = 0.4 * (pitch_mean / 300)  # normalized by ~max pitch expected
    hand_score = 0.35 * hand_ratio
    face_score = 0.25 * ((emotion_ratio + head_ratio) / 2)

    confidence = voice_score + hand_score + face_score

    # Boost confidence by 10%, cap at 100%
    boosted_confidence = min(round(confidence * 100 ), 100)

    st.subheader(f"💡 Your Confidence Score: **{boosted_confidence}%**")

    st.write(f"**Debug info:**")
    st.write(f"- Smile ratio: {emotion_ratio:.2f}")
    st.write(f"- Head centered ratio: {head_ratio:.2f}")
    st.write(f"- Hand activity ratio: {hand_ratio:.2f}")
    st.write(f"- Mean pitch: {pitch_mean:.2f} Hz")
    st.write(f"- Pitch std deviation: {pitch_std:.2f}")

    if boosted_confidence > 85:
        st.success("🔥 You nailed it! Amazing energy and clarity.")
    elif boosted_confidence > 60:
        st.info("😊 You're doing great! Keep practicing.")
    else:
        st.warning("💪 Good start. Try adding more hand movement and vocal energy.")

    # Save logs
    data = {
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Confidence_Score": [boosted_confidence],
        "Smile_Ratio": [round(emotion_ratio, 3)],
        "Head_Center_Ratio": [round(head_ratio, 3)],
        "Hand_Activity_Ratio": [round(hand_ratio, 3)],
        "Pitch_Mean_Hz": [round(pitch_mean, 2)],
        "Pitch_Std": [round(pitch_std, 2)]
    }
    df_new = pd.DataFrame(data)
    try:
        df_old = pd.read_csv("confidence_log.csv")
        df = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df = df_new
    df.to_csv("confidence_log.csv", index=False)
    st.info("📁 Session saved to 'confidence_log.csv'")

    # Show trend
    if len(df) > 1:
        st.subheader("📈 Confidence Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Confidence_Score'], marker='o')
        plt.xticks(rotation=45, ha='right')
        ax.set_ylabel("Confidence Score")
        st.pyplot(fig)
    else:
        st.write("Try more sessions to build your confidence trend!")

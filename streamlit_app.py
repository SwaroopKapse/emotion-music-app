import streamlit as st
import cv2
from deepface import DeepFace
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
import sounddevice as sd
from scipy.io.wavfile import write

# ---------- Spotify Config ----------
SPOTIFY_CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
SPOTIFY_CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"

emotion_map = {
    "happy": "happy hits",
    "sad": "sad songs",
    "angry": "angry rock",
    "fear": "chill",
    "surprise": "upbeat songs",
    "neutral": "calm instrumental"
}

# ---------- Emotion from Image ----------
def detect_face_emotion(image):
    with open("temp.jpg", "wb") as f:
        f.write(image.read())
    result = DeepFace.analyze(img_path="temp.jpg", actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']

# ---------- Emotion from Text ----------
def get_text_emotion(text):
    classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
    result = classifier(text)
    return result[0]['label'].lower()

# ---------- Record Voice ----------
def record_voice():
    fs = 44100
    duration = 5
    st.info("Recording voice for 5 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write('voice.wav', fs, recording)
    st.success("Voice recorded!")
    return "sad"  # Replace with actual model later

# ---------- Spotify Music Player ----------
def play_spotify(emotion):
    auth = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=auth)
    query = emotion_map.get(emotion, "mood")
    results = sp.search(q=query, type="playlist", limit=1)
    url = results['playlists']['items'][0]['external_urls']['spotify']
    st.markdown(f"[Click here to open Spotify Playlist for **{emotion.title()} Mood** 🎵]({url})", unsafe_allow_html=True)
    webbrowser.open(url)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Emotion Music Recommender", layout="centered")
st.title("🎧 Emotion-Based Music Recommender")
st.write("Detect your emotion and get the perfect playlist 💖")

mode = st.radio("Select Emotion Input Method:", ["😊 Face", "🎤 Voice", "📝 Text"])

if mode == "😊 Face":
    image = st.file_uploader("Upload a clear photo", type=["jpg", "jpeg", "png"])
    if st.button("Detect Emotion"):
        if image:
            emotion = detect_face_emotion(image)
            st.success(f"Emotion Detected: **{emotion.upper()}**")
            play_spotify(emotion)
        else:
            st.error("Please upload an image.")

elif mode == "🎤 Voice":
    if st.button("Record & Detect Emotion"):
        emotion = record_voice()
        st.success(f"Emotion Detected: **{emotion.upper()}** (Demo)")
        play_spotify(emotion)

elif mode == "📝 Text":
    text_input = st.text_input("Type something about how you feel")
    if st.button("Analyze Text"):
        if text_input:
            emotion = get_text_emotion(text_input)
            st.success(f"Emotion Detected: **{emotion.upper()}**")
            play_spotify(emotion)
        else:
            st.error("Please enter some text.")


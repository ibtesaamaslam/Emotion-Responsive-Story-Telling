import cv2
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import time

# Placeholder for wearable device integration (heart rate)
class WearableDevice:
    def get_heart_rate(self):
        # Simulate heart rate data (60-120 bpm)
        return random.randint(60, 120)

# Emotion detection class
class EmotionDetector:
    def __init__(self):
        self.wearable = WearableDevice()

    def detect_facial_emotion(self, frame):
        # Placeholder: Simulate emotion based on image brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray)
        if intensity > 150:
            return {"happy": 0.7, "sad": 0.1, "angry": 0.1, "fear": 0.05, "neutral": 0.05}
        elif intensity < 100:
            return {"happy": 0.1, "sad": 0.7, "angry": 0.1, "fear": 0.05, "neutral": 0.05}
        else:
            return {"happy": 0.2, "sad": 0.2, "angry": 0.2, "fear": 0.2, "neutral": 0.2}

    def detect_voice_sentiment(self):
        # Placeholder: Simulate voice sentiment without audio input
        return random.choice(["excited", "calm"])

    def get_physiological_emotion(self):
        hr = self.wearable.get_heart_rate()
        return "stressed" if hr > 90 else "relaxed"

    def combine_emotions(self, facial, voice, physio):
        weights = {"facial": 0.5, "voice": 0.3, "physio": 0.2}
        dominant_facial = max(facial, key=facial.get)
        return {
            "dominant_emotion": dominant_facial,
            "confidence": facial[dominant_facial] * weights["facial"]
            + (1 if voice == "excited" else 0.5) * weights["voice"]
            + (1 if physio == "stressed" else 0.5) * weights["physio"],
        }

# Narrative generation class
class StoryGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.genres = {
            "happy": "adventure",
            "sad": "drama",
            "angry": "epic",
            "fear": "mystery",
            "neutral": "fantasy",
        }

    def generate_segment(self, prompt, emotion, user_input=None):
        genre = self.genres.get(emotion, "fantasy")
        full_prompt = f"In a {genre} story, {prompt}"
        if user_input:
            full_prompt += f" Incorporate: {user_input}"
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=100)
        outputs = self.model.generate(
            inputs["input_ids"], max_length=150, num_return_sequences=1, no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Visual rendering class
class VisualRenderer:
    def render(self, emotion, story_segment):
        colors = {
            "happy": (255, 255, 0),  # Yellow
            "sad": (0, 0, 255),  # Blue
            "angry": (255, 0, 0),  # Red
            "fear": (128, 0, 128),  # Purple
            "neutral": (255, 255, 255),  # White
        }
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = colors.get(emotion, (255, 255, 255))
        return img

# Main application
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    emotion_detector = EmotionDetector()
    story_generator = StoryGenerator()
    visual_renderer = VisualRenderer()
    story_prompt = "A hero embarks on a journey."
    story_segments = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect emotions
        facial_emotions = emotion_detector.detect_facial_emotion(frame)
        voice_sentiment = emotion_detector.detect_voice_sentiment()
        physio_state = emotion_detector.get_physiological_emotion()
        emotion_data = emotion_detector.combine_emotions(facial_emotions, voice_sentiment, physio_state)

        # Generate story segment
        user_input = input("Add to story (or press Enter): ") or None
        segment = story_generator.generate_segment(
            story_prompt, emotion_data["dominant_emotion"], user_input
        )
        story_segments.append(segment)
        story_prompt = segment[-50:]  # Update prompt with latest segment

        # Render visual
        visual = visual_renderer.render(emotion_data["dominant_emotion"], segment)

        # Display output
        cv2.imshow("Emotion-Responsive Storytelling", frame)
        cv2.imshow("Story Visual", visual)
        print(f"Story Segment: {segment}")

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
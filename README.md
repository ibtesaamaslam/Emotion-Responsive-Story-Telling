
# Emotion-Responsive Storytelling

An innovative system that dynamically generates narrative segments and visual outputs based on **real-time emotional inputs** from the user. By combining **facial emotion detection**, **voice sentiment analysis**, and **simulated heart rate data**, it creates interactive storytelling experiences that evolve with your emotions.

---

## 🔥 Features

* **Emotion Detection**
  Uses webcam-based facial analysis, voice sentiment detection, and simulated heart rate to infer dominant user emotion.

* **Dynamic Storytelling**
  Generates story segments using GPT-2, with genres like adventure, drama, mystery, fantasy, and epic—tailored to your emotional state.

* **Visual Rendering**
  Creates color-coded visuals that match the detected emotion to enhance immersion.

* **User Interaction**
  Lets users influence the story by providing optional text inputs at runtime.

* **Modular Architecture**
  Clean, class-based structure:

  * `EmotionDetector`
  * `StoryGenerator`
  * `VisualRenderer`
    ...easy to extend and maintain.

---

## 🧠 Technologies Used

**Python Libraries**

* `OpenCV` for webcam capture and image analysis
* `NumPy` for image and data processing
* `PyTorch`, `Transformers` (Hugging Face) for GPT-2 generation
* `Librosa` (placeholder) for voice sentiment input
* `Matplotlib`, `PIL` for rendering visuals

**Hardware**

* Requires a webcam for facial detection
* Simulated heart rate via a mock `WearableDevice` class

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/emotion-responsive-storytelling.git
   cd emotion-responsive-storytelling
   ```

2. **Install Dependencies**
   Make sure Python 3.8+ is installed.

   ```bash
   pip install opencv-python numpy torch transformers librosa matplotlib pillow
   ```

   > `pyaudio` is optional and not required for this version. It’s excluded for compatibility with platforms like Google Colab.

3. **Run the App**

   ```bash
   python emotion_responsive_storytelling.py
   ```

   Or run the Jupyter notebook:

   ```bash
   open Emotion_Responsive_Storytelling.ipynb
   ```

---

## 💡 How It Works

* Captures webcam feed to simulate **facial emotion detection** (based on image brightness).
* Simulates **voice sentiment** and **heart rate** (e.g., 60–120 bpm).
* Infers a dominant emotion: `happy`, `sad`, `angry`, `fear`, or `neutral`.
* Generates a story segment using **GPT-2**, aligned with that emotion’s narrative genre.
* Displays:

  * A live webcam feed
  * A 200x200 color-coded image (e.g., yellow for happy)
  * The story in the console
* Prompts the user to optionally input text to influence the next scene.

---

## 🧪 Example Output

**Story Segment:**

> In a fantasy story, the hero embarks on a journey. They discover a hidden valley filled with enchanted creatures...

**Visual Output:**

> A square of glowing yellow (for "happy"), displayed in a new window.

---

## ⚠️ Limitations

* **Simulated Emotion Detection**: Facial emotion is based on brightness; voice sentiment is randomly assigned.
* **Simulated Heart Rate**: Heart rate data is fake; needs wearable integration.
* **Hardware Dependency**: Needs a working webcam.
* **Performance**: GPT-2 may lag on machines without GPU acceleration.

---

## 🚀 Future Enhancements

* Replace facial detection with DeepFace or fer2013-based models
* Use real voice sentiment models like VGGish
* Connect to Bluetooth-enabled wearables for live heart rate
* Render AI-generated visuals using DALL·E or VQ-VAE
* Add narration using `pyttsx3` or another text-to-speech engine

---

## 🤝 Contributing

Contributions welcome!
To contribute:

1. Open an issue to discuss ideas
2. Fork the repo, create a branch, and submit a PR
3. Follow PEP 8 for coding style

---

## 📄 License

**MIT License 2.0**
See `LICENSE` for full details.

---

## 🙏 Acknowledgments

* Inspired by research in **affective computing** and **interactive storytelling**
* GPT-2 via Hugging Face Transformers
* Thanks to open-source communities behind OpenCV, Librosa, and more


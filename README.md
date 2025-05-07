# SpeechToText
A speech-to-text transcription model built using OpenAI's Whisper, supporting Hindi, Marathi, and English. The model offers an interactive and fun transcription experience, with humorous prompts and real-time updates. It also includes Text-to-Speech (TTS) for spoken transcription output.


# Speech Transcriber with Whisper and PyAudio

This project records speech from your microphone, saves it as a `.wav` file, and then transcribes it using OpenAI's Whisper model. It supports transcription in multiple languages and is built to run smoothly using GPU acceleration with PyTorch.

---

## ⚙️ Setting Up the Project (Must-Do)

### 1. Create a Virtual Environment (VERY IMPORTANT)

Avoid version conflicts by setting up a virtual environment:

```powershell
# If you don’t have admin privileges:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create virtual environment
python -m venv your_env_name

# Activate the environment (PowerShell)
.\your_env_name\Scripts\Activate

# Select Python interpreter in VS Code:
Ctrl + Shift + P → "Python: Select Interpreter" → Select your venv interpreter

# To deactivate:
deactivate
```

List installed packages:

```bash
python -m pip list
```

Install new packages:

```bash
python -m pip install package_name
```

---

## 🎤 Step 1: Record Audio Using PyAudio

### How It Works

* Microphones convert sound waves into voltage distributions (bits).
* These bits are captured in real-time using `PyAudio`, stored in chunks (frames), and later joined to form a complete audio.

### Setup Stream

```python
p = pyaudio.PyAudio()
stream = p.open(
  rate=44100,
  frames_per_buffer=1024,
  format=pyaudio.paInt16,
  input=True,
  channels=1
)
```

### Recording Loop

```python
frames = []
for _ in range(0, 215):  # 43 frames/sec * 5 seconds
    data = stream.read(1024)
    frames.append(data)
```

---

## 💾 Step 2: Save to `.wav` Using `wave` Module

```python
wf = wave.open("output.wav", 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(44100)
wf.writeframes(b''.join(frames))
wf.close()
```

*The `wave` module is built-in with Python and does not need to be installed.*

---

## 🧠 Step 3: Install and Use `speech_recognition` (Initially Used)

Although Google’s SpeechRecognition API works, it is limited to English. We switched to Whisper for better multi-language support.

---

## 🌀 Step 4: Use Whisper for Speech Recognition

### Install Whisper and Dependencies

```bash
python -m pip install git+https://github.com/openai/whisper.git
python -m pip install ffmpeg-python
```

### Load Model and Transcribe

```python
import whisper
model = whisper.load_model("medium").to("cuda")
result = model.transcribe("output.wav")
print(result["text"])
```

### Model Types:

* `'base'`: Lightweight but less accurate
* `'medium'`: Good balance, needs GPU (CUDA cores recommended)
* `'large'`: Best accuracy, resource-intensive

### Output Dictionary:

* `"text"`: Transcribed text
* `"language"`: Detected language (e.g., "hi" for Hindi)
* `"segments"`: Not required for basic use

---

## 🔥 Step 5: Install PyTorch with CUDA for GPU Acceleration

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Why PyTorch?

Whisper is ML-based and benefits from CUDA acceleration on NVIDIA GPUs. This setup ensures faster transcription.

---

## ✅ Final Output

* `.wav` file saved locally
* Text transcription printed or saved
* GPU-accelerated for fast processing

---

## 🚀 Future Ideas

* Add UI for language selection
* Auto language detection with fallback
* Export transcription as `.txt` or subtitle formats

---

## 📂 Project Structure (Sample)

```
📁 TRANSCRIBINGMODEL
├── Speechrecognition.py
├── output.wav
├── requirements.txt
└── README.md
```

### 📦 Step: Install All Requirements

Install all dependencies at once using:

```bash
pip install -r requirements.txt
---


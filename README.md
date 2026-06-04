# Real-time-conversational-voice-assistant

A real-time AI voice assistant powered by OpenAI that listens, remembers conversations, speaks naturally, and supports interruption detection during responses.

## Features

* 🎙️ Real-time Speech-to-Text using Whisper
* 🧠 Context-aware conversation memory
* 💬 GPT-powered conversational responses
* ⚡ Streaming response generation
* 🔊 Streaming Text-to-Speech playback
* ✋ Adaptive interruption detection (barge-in support)
* 🎚️ Automatic voice calibration
* 🤫 Smart silence detection
* 🧵 Multi-threaded processing pipeline
* 🚀 Low-latency conversational experience

---

## How It Works

```text
User Speech
    ↓
Voice Activity Detection
    ↓
Whisper Transcription
    ↓
GPT Response Generation
    ↓
Streaming TTS
    ↓
Audio Playback
    ↓
User Can Interrupt Anytime
```

---

## Tech Stack

* Python
* OpenAI API
* Whisper (Speech-to-Text)
* GPT-4o-mini
* OpenAI TTS
* SoundDevice
* NumPy
* PyDub
* FFmpeg
* Wavio

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/echonova.git
cd echonova
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

Download FFmpeg and ensure it is available on your system path.

[https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

Update the FFmpeg path in the script if required.

### 4. Create Environment File

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Required Packages

```bash
pip install openai
pip install sounddevice
pip install wavio
pip install numpy
pip install python-dotenv
pip install pydub
```

Or simply:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

Example:

```text
🎙️ Listening...

You: What's the weather today?

🤖 Assistant:
Today's weather is sunny with a high of 28°C.
```

---

## Core Features

### Conversation Memory

The assistant stores recent conversation history and sends it with each request, allowing it to maintain context naturally across multiple exchanges.

### Streaming Responses

Responses are generated token-by-token, enabling speech synthesis to start before the entire answer is complete.

### Adaptive Interruption Detection

While speaking, the assistant continuously monitors microphone input.

If the user starts speaking:

* Audio playback stops immediately
* Ongoing generation is canceled
* The assistant begins listening again

### Smart Voice Calibration

The system learns the user's speaking volume over time and dynamically adjusts interruption sensitivity for improved accuracy.

### Automatic Silence Detection

Recording automatically stops when speech ends, creating a hands-free conversational experience.

---

## Architecture

```text
┌──────────────┐
│ User Speech  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Audio Input  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Whisper STT  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ GPT-4o-mini  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ OpenAI TTS   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Audio Output │
└──────────────┘
```

---

## Project Structure

```text
.
├── main.py
├── .env
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Wake-word activation
* OpenAI Realtime API integration
* Long-term memory storage
* GUI/Desktop application
* Multi-language support
* Speaker recognition
* Emotion-aware responses
* Mobile deployment

---

## License

MIT License

---

## Acknowledgements

Built using:

* OpenAI
* Whisper
* PyDub
* FFmpeg
* SoundDevice
* NumPy


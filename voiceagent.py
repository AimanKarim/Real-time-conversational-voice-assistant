import os
import sounddevice as sd
import wavio
import tempfile
import io
import threading
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
import time
import queue

# ✅ Load API Key
load_dotenv(r"C:\Users\NPRD-MISHAAL\Downloads\VoiceAgent\secret.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Set FFmpeg paths
ffmpeg_path = r"C:\Users\NPRD-MISHAAL\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")
os.environ["PATH"] += os.pathsep + ffmpeg_path

# 🎙️ Global state
class AssistantState:
    def __init__(self):
        self.is_speaking = False
        self.should_stop_speaking = False
        self.interrupt_cooldown = 0
        self.stop_all_processing = False
        self.lock = threading.Lock()
        self.user_voice_level = 0.04
        self.background_noise_level = 0.01
        self.conversation_history = []  # Store conversation memory
    
    def reset(self):
        with self.lock:
            self.is_speaking = False
            self.should_stop_speaking = False
            self.stop_all_processing = False
    
    def trigger_interruption(self):
        with self.lock:
            self.should_stop_speaking = True
            self.stop_all_processing = True
    
    def calibrate_voice(self, level):
        with self.lock:
            # Use exponential moving average to track voice level
            if self.user_voice_level == 0.04:  # First calibration
                self.user_voice_level = level
            else:
                # Blend old and new (70% old, 30% new) to adapt over time
                self.user_voice_level = 0.7 * self.user_voice_level + 0.3 * level
    
    def update_background(self, level):
        with self.lock:
            self.background_noise_level = 0.9 * self.background_noise_level + 0.1 * level
    
    def add_to_history(self, role, content):
        """Add message to conversation history"""
        with self.lock:
            self.conversation_history.append({"role": role, "content": content})
            # Keep only last 10 messages (5 exchanges) to avoid token limits
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
    
    def get_messages(self, user_text):
        """Get messages for GPT API including history"""
        with self.lock:
            messages = [
                {"role": "system", "content": "You are a helpful English assistant. Keep responses concise and conversational. Remember context from previous messages."}
            ]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_text})
            return messages

state = AssistantState()

# 🎙️ Record audio with VAD and voice calibration
def record_audio_smart(max_duration=10, samplerate=16000, silence_threshold=0.015, silence_duration=1.2):
    """Records audio and stops when silence is detected"""
    
    timeout = 0
    while state.is_speaking and timeout < 50:
        time.sleep(0.1)
        timeout += 1
    
    if timeout >= 50:
        state.reset()
    
    print("🎙️ Listening...")
    
    recording = []
    silent_chunks = 0
    speech_detected = False
    chunks_per_second = int(samplerate / 1024)
    max_silent_chunks = int(silence_duration * chunks_per_second)
    
    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype='int16')
    stream.start()
    
    max_level_during_speech = 0
    
    for _ in range(int(max_duration * chunks_per_second)):
        chunk, _ = stream.read(1024)
        audio_level = np.abs(chunk).mean() / 32768
        
        if audio_level > silence_threshold and not speech_detected:
            speech_detected = True
        
        if speech_detected and audio_level > max_level_during_speech:
            max_level_during_speech = audio_level
        
        recording.append(chunk)
        
        if speech_detected:
            if audio_level < silence_threshold:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    break
            else:
                silent_chunks = 0
    
    stream.stop()
    stream.close()
    
    # Calibrate user's voice level
    if max_level_during_speech > 0.02:
        state.calibrate_voice(max_level_during_speech)
    
    recording = np.concatenate(recording, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(tmp.name, recording, samplerate, sampwidth=2)
    return tmp.name

# 🎧 Transcribe audio → text
def transcribe_audio(filename):
    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    return transcript.text.strip()

# 💬 GPT reply with streaming AND memory
def generate_reply_streaming(text_queue, user_text):
    """Stream GPT response with conversation memory"""
    try:
        # Get messages including conversation history
        messages = state.get_messages(user_text)
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        buffer = ""
        full_response = ""  # Track full response for memory
        sentence_endings = {'.', '!', '?'}
        
        for chunk in stream:
            if state.stop_all_processing:
                break
                
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer += content
                full_response += content
                print(content, end="", flush=True)
                
                for ending in sentence_endings:
                    if ending in buffer:
                        parts = buffer.split(ending, 1)
                        sentence = parts[0] + ending
                        buffer = parts[1] if len(parts) > 1 else ""
                        
                        if sentence.strip():
                            text_queue.put(sentence.strip())
                        break
        
        if buffer.strip() and not state.stop_all_processing:
            text_queue.put(buffer.strip())
        
        # Add assistant's response to memory (if not interrupted)
        if not state.stop_all_processing and full_response.strip():
            state.add_to_history("assistant", full_response.strip())
        
        text_queue.put(None)
        if not state.stop_all_processing:
            print()
            
    except Exception as e:
        if not state.stop_all_processing:
            print(f"\n❌ Error: {e}")
        text_queue.put(None)

# 🔊 TTS generator
def generate_audio_chunks(text_queue, audio_queue):
    """Generate audio for each text chunk and put in audio queue"""
    try:
        while not state.stop_all_processing:
            try:
                text = text_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            
            if text is None or state.stop_all_processing:
                audio_queue.put(None)
                break
            
            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                    response_format="mp3"
                )
                
                audio_data = response.read()
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                samples = samples.astype(np.float32) / (2**15)
                
                if not state.stop_all_processing:
                    audio_queue.put((samples, audio.frame_rate))
            except Exception as e:
                if not state.stop_all_processing:
                    print(f"\n❌ Error: {e}")
    
    except Exception as e:
        if not state.stop_all_processing:
            print(f"\n❌ Error: {e}")
    finally:
        audio_queue.put(None)

# 🔊 Audio playback
def play_audio_stream(audio_queue):
    """Play audio chunks as they arrive with no gaps"""
    
    state.is_speaking = True
    state.should_stop_speaking = False
    state.stop_all_processing = False
    state.interrupt_cooldown = time.time() + 0.2  # Shorter cooldown
    
    try:
        playing = False
        
        while not state.stop_all_processing:
            try:
                chunk_data = audio_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            
            if chunk_data is None:
                break
            
            if state.stop_all_processing:
                break
            
            samples, sample_rate = chunk_data
            
            if playing:
                while True:
                    if state.stop_all_processing:
                        sd.stop()
                        break
                    try:
                        if not sd.get_stream().active:
                            break
                    except:
                        break
                    time.sleep(0.01)
            
            if state.stop_all_processing:
                break
            
            sd.play(samples, sample_rate, blocking=False)
            playing = True
        
        if not state.stop_all_processing and playing:
            try:
                sd.wait()
            except:
                pass
    
    except Exception as e:
        if not state.stop_all_processing:
            print(f"\n❌ Error: {e}")
    
    finally:
        try:
            sd.stop()
        except:
            pass
        state.is_speaking = False

# 🎙️ MORE SENSITIVE interruption detection
def listen_for_interruption(samplerate=16000, confirmation_chunks=2):
    """
    More sensitive adaptive threshold
    """
    try:
        stream = sd.InputStream(samplerate=samplerate, channels=1, dtype='int16')
        stream.start()
        
        consecutive_triggers = 0
        
        # MORE AGGRESSIVE: 40% of user's voice (was 50%)
        # AND lower background multiplier: 2x (was 2.5x)
        dynamic_threshold = max(
            state.user_voice_level * 0.4,  # Lower percentage = more sensitive
            state.background_noise_level * 2.0  # Lower multiplier = more sensitive
        )
        
        while state.is_speaking and not state.stop_all_processing:
            if time.time() < state.interrupt_cooldown:
                time.sleep(0.05)
                consecutive_triggers = 0
                continue
            
            chunk, _ = stream.read(1024)
            audio_level = np.abs(chunk).mean() / 32768
            
            # Update background noise estimate
            if audio_level < dynamic_threshold * 0.5:
                state.update_background(audio_level)
            
            # Check if above threshold
            if audio_level > dynamic_threshold:
                consecutive_triggers += 1
                
                if consecutive_triggers >= confirmation_chunks:
                    print(f"\n🛑 Interrupted")
                    
                    state.trigger_interruption()
                    sd.stop()
                    break
            else:
                if consecutive_triggers > 0:
                    consecutive_triggers = 0
        
        stream.stop()
        stream.close()
            
    except Exception as e:
        if not state.stop_all_processing:
            print(f"\n❌ Error: {e}")

# 🔊 Parallel generation and speaking
def generate_and_speak(user_text):
    """Generate text, audio, and speak simultaneously"""
    
    text_queue = queue.Queue(maxsize=10)
    audio_queue = queue.Queue(maxsize=5)
    
    state.reset()
    
    t1 = threading.Thread(target=generate_reply_streaming, args=(text_queue, user_text), daemon=True)
    t2 = threading.Thread(target=generate_audio_chunks, args=(text_queue, audio_queue), daemon=True)
    t3 = threading.Thread(target=play_audio_stream, args=(audio_queue,), daemon=True)
    t4 = threading.Thread(target=listen_for_interruption, daemon=True)
    
    threads = [t1, t2, t3, t4]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join(timeout=30)
    
    while not text_queue.empty():
        try:
            text_queue.get_nowait()
        except:
            break
    
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break
    
    time.sleep(0.1)
    state.reset()

# 🔁 Main loop
STOP_WORDS = {"stop", "quit", "exit", "bye", "goodbye"}

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Voice Assistant with Memory!")
    print("=" * 60)
    print("💡 Features:")
    print("   - Remembers your conversation")
    print("   - Adaptive interruption (learns your voice)")
    print("   - Say 'stop', 'quit', 'exit', or 'bye' to end")
    print("   - Use headphones for best results")
    print("=" * 60)
    print()
    
    while True:
        try:
            state.reset()
            sd.stop()
            
            audio_file = record_audio_smart(max_duration=10, silence_duration=1.2)
            user_text = transcribe_audio(audio_file)
            
            try:
                os.unlink(audio_file)
            except:
                pass
            
            print(f"✅ You: {user_text}")
            
            # Add user message to history
            state.add_to_history("user", user_text)
            
            if any(word in user_text.lower() for word in STOP_WORDS):
                print("👋 Goodbye!")
                state.is_speaking = True
                
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input="Goodbye! It was nice talking to you.",
                    response_format="mp3"
                )
                audio_data = response.read()
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                samples = samples.astype(np.float32) / (2**15)
                sd.play(samples, audio.frame_rate)
                sd.wait()
                break
            
            print("🤖 Assistant: ", end="", flush=True)
            generate_and_speak(user_text)
            
            if state.stop_all_processing:
                time.sleep(0.2)
            else:
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            state.reset()
            sd.stop()
            time.sleep(0.5)
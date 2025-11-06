# =====================================================
# VOICE TRANSLATION TOOL - COMPLETE PROJECT
# =====================================================

"""
PROJECT STRUCTURE:
voice_translation_tool/
│
├── app.py                 # Main Streamlit UI
├── converter.py           # Audio format conversion
├── speech_to_text.py      # Speech recognition
├── translator.py          # Text translation
├── requirements.txt       # Dependencies
├── README.md             # Setup instructions
└── temp/                 # Temporary files (auto-created)

INSTALLATION COMMANDS:
pip install streamlit
pip install pydub
pip install SpeechRecognition
pip install googletrans==4.0.0-rc1
pip install openai-whisper
pip install pyaudio
pip install sounddevice
pip install soundfile
pip install numpy

# For ffmpeg (required for audio conversion):
# Windows: Download from https://ffmpeg.org/download.html
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg
"""

# =====================================================
# FILE 1: converter.py - Audio Format Conversion
# =====================================================

import os
from pydub import AudioSegment
import tempfile

class AudioConverter:
    """Handles audio/video format conversion to WAV"""
    
    SUPPORTED_FORMATS = ['.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.flac', '.aac']
    
    def __init__(self, temp_dir='temp'):
        self.temp_dir = temp_dir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    def is_supported(self, filename):
        """Check if file format is supported"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.SUPPORTED_FORMATS
    
    def convert_to_wav(self, input_file, output_path=None):
        """
        Convert any supported audio/video format to WAV
        
        Args:
            input_file: Path to input file
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to converted WAV file
        """
        try:
            # Get file extension
            ext = os.path.splitext(input_file)[1].lower()
            
            # If already WAV, return as is
            if ext == '.wav':
                return input_file
            
            # Generate output path if not provided
            if output_path is None:
                output_path = os.path.join(
                    self.temp_dir, 
                    f"converted_{os.path.basename(input_file).rsplit('.', 1)[0]}.wav"
                )
            
            # Load audio/video file
            if ext == '.mp4':
                audio = AudioSegment.from_file(input_file, format='mp4')
            elif ext == '.mp3':
                audio = AudioSegment.from_file(input_file, format='mp3')
            elif ext == '.m4a':
                audio = AudioSegment.from_file(input_file, format='m4a')
            elif ext == '.ogg':
                audio = AudioSegment.from_file(input_file, format='ogg')
            elif ext == '.flac':
                audio = AudioSegment.from_file(input_file, format='flac')
            elif ext == '.aac':
                audio = AudioSegment.from_file(input_file, format='aac')
            else:
                audio = AudioSegment.from_file(input_file)
            
            # Export as WAV
            audio.export(output_path, format='wav')
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Conversion error: {str(e)}")
    
    def get_audio_info(self, file_path):
        """Get audio file information"""
        try:
            audio = AudioSegment.from_file(file_path)
            return {
                'duration': len(audio) / 1000.0,  # seconds
                'channels': audio.channels,
                'sample_rate': audio.frame_rate,
                'sample_width': audio.sample_width
            }
        except:
            return None


# =====================================================
# FILE 2: speech_to_text.py - Speech Recognition
# =====================================================

import speech_recognition as sr
import os

class SpeechToText:
    """Handles speech-to-text conversion using multiple engines"""
    
    def __init__(self, engine='google'):
        """
        Initialize speech recognition
        
        Args:
            engine: 'google' (online) or 'whisper' (offline)
        """
        self.engine = engine
        self.recognizer = sr.Recognizer()
    
    def transcribe_google(self, audio_file, language='en-US'):
        """
        Transcribe using Google Speech Recognition (online)
        
        Args:
            audio_file: Path to WAV file
            language: Language code (e.g., 'en-US', 'hi-IN', 'mr-IN')
            
        Returns:
            Transcribed text
        """
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio
                audio_data = self.recognizer.record(source)
                
                # Recognize speech
                text = self.recognizer.recognize_google(
                    audio_data, 
                    language=language
                )
                
                return text
                
        except sr.UnknownValueError:
            raise Exception("Could not understand audio. Please speak clearly.")
        except sr.RequestError as e:
            raise Exception(f"Google API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")
    
    def transcribe_whisper(self, audio_file, language=None):
        """
        Transcribe using OpenAI Whisper (offline)
        
        Args:
            audio_file: Path to audio file
            language: Optional language code (e.g., 'en', 'hi', 'mr')
            
        Returns:
            Transcribed text
        """
        try:
            import whisper
            
            # Load Whisper model (base by default)
            model = whisper.load_model("base")
            
            # Transcribe
            if language:
                result = model.transcribe(audio_file, language=language)
            else:
                result = model.transcribe(audio_file)
            
            return result['text'].strip()
            
        except ImportError:
            raise Exception("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            raise Exception(f"Whisper transcription error: {str(e)}")
    
    def transcribe(self, audio_file, language='en-US'):
        """
        Main transcription method
        
        Args:
            audio_file: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text
        """
        if self.engine == 'google':
            return self.transcribe_google(audio_file, language)
        elif self.engine == 'whisper':
            # Convert language code for Whisper (e.g., 'en-US' -> 'en')
            whisper_lang = language.split('-')[0] if '-' in language else language
            return self.transcribe_whisper(audio_file, whisper_lang)
        else:
            raise Exception(f"Unknown engine: {self.engine}")
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return {
            'English (US)': 'en-US',
            'English (UK)': 'en-GB',
            'Hindi': 'hi-IN',
            'Marathi': 'mr-IN',
            'Bengali': 'bn-IN',
            'Tamil': 'ta-IN',
            'Telugu': 'te-IN',
            'Gujarati': 'gu-IN',
            'Kannada': 'kn-IN',
            'Malayalam': 'ml-IN',
            'Punjabi': 'pa-IN',
            'Spanish': 'es-ES',
            'French': 'fr-FR',
            'German': 'de-DE',
            'Italian': 'it-IT',
            'Japanese': 'ja-JP',
            'Korean': 'ko-KR',
            'Chinese (Mandarin)': 'zh-CN',
            'Arabic': 'ar-SA',
            'Russian': 'ru-RU',
            'Portuguese': 'pt-BR'
        }


# =====================================================
# FILE 3: translator.py - Text Translation
# =====================================================

from deep_translator import GoogleTranslator

class TextTranslator:
    """Handles text translation using Google Translate"""
    
    def __init__(self):
        self.translator = GoogleTranslator()
    
    def translate(self, text, target_language='en', source_language='auto'):
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'en', 'hi', 'mr')
            source_language: Source language code ('auto' for auto-detect)
            
        Returns:
            Translated text
        """
        try:
            if not text or text.strip() == '':
                return ''
            
            result = GoogleTranslator(
                source=source_language,
                target=target_language
            ).translate(text)
            
            return result
            
        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            from langdetect import detect
            lang = detect(text)
            return {
                'language': lang,
                'confidence': 0.99
            }
        except:
            return None
    
    def get_supported_languages(self):
        """Get all supported languages"""
        return {
            'English': 'en',
            'Hindi': 'hi',
            'Marathi': 'mr',
            'Bengali': 'bn',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Gujarati': 'gu',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Punjabi': 'pa',
            'Urdu': 'ur',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Chinese (Simplified)': 'zh-CN',
            'Chinese (Traditional)': 'zh-TW',
            'Arabic': 'ar',
            'Russian': 'ru',
            'Turkish': 'tr',
            'Dutch': 'nl',
            'Swedish': 'sv',
            'Polish': 'pl',
            'Indonesian': 'id',
            'Vietnamese': 'vi',
            'Thai': 'th'
        }


# =====================================================
# FILE 4: app.py - Streamlit UI
# =====================================================

import streamlit as st
import os
import tempfile
from datetime import datetime
import traceback

# Initialize components
@st.cache_resource
def init_components():
    converter = AudioConverter()
    translator = TextTranslator()
    return converter, translator

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory"""
    try:
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def record_audio():
    """Record audio from microphone"""
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        
        duration = st.session_state.get('record_duration', 5)
        sample_rate = 44100
        
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        # Save to file
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        file_path = os.path.join(temp_dir, filename)
        
        sf.write(file_path, audio_data, sample_rate)
        
        st.success("✅ Recording complete!")
        return file_path
        
    except ImportError:
        st.error("Microphone recording requires: pip install sounddevice soundfile numpy")
        return None
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Voice Translation Tool",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Initialize components
    converter, translator = init_components()
    
    # Header
    st.title("🎙️ Voice Translation Tool")
    st.markdown("### Convert speech to text and translate to any language")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Engine selection
        engine = st.selectbox(
            "Speech Recognition Engine",
            ['Google (Online)', 'Whisper (Offline)'],
            help="Google: Fast, online. Whisper: Accurate, works offline."
        )
        engine_code = 'google' if 'Google' in engine else 'whisper'
        
        # Source language
        stt = SpeechToText(engine=engine_code)
        source_langs = stt.get_supported_languages()
        source_lang_name = st.selectbox(
            "Source Language (Audio)",
            list(source_langs.keys()),
            index=0
        )
        source_lang_code = source_langs[source_lang_name]
        
        # Translation option
        translate_option = st.checkbox("Translate Text", value=True)
        
        if translate_option:
            target_langs = translator.get_supported_languages()
            target_lang_name = st.selectbox(
                "Target Language (Translation)",
                list(target_langs.keys()),
                index=1  # Hindi by default
            )
            target_lang_code = target_langs[target_lang_name]
        
        st.markdown("---")
        st.subheader("📊 Supported Formats")
        st.info("MP3, MP4, WAV, M4A, OGG, FLAC, AAC")
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        **Features:**
        - Upload audio/video files
        - Record from microphone
        - Speech-to-text conversion
        - Multi-language translation
        - Download results
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Record Audio"])
    
    # Tab 1: File Upload
    with tab1:
        st.subheader("Upload Audio/Video File")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac', 'aac'],
            help="Upload audio or video file for transcription"
        )
        
        if uploaded_file:
            # Display file info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**File:** {uploaded_file.name}")
            with col2:
                st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            # Save file
            file_path = save_uploaded_file(uploaded_file)
            
            if file_path:
                # Convert & Translate button
                if st.button("🚀 Convert & Translate", type="primary", key="upload_btn"):
                    process_audio(
                        file_path,
                        stt,
                        translator,
                        converter,
                        source_lang_code,
                        target_lang_code if translate_option else None,
                        source_lang_name,
                        target_lang_name if translate_option else None
                    )
    
    # Tab 2: Record Audio
    with tab2:
        st.subheader("Record from Microphone")
        
        # Recording duration
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=3,
            max_value=60,
            value=10,
            step=1
        )
        st.session_state['record_duration'] = duration
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎤 Start Recording", type="primary"):
                st.session_state['recording_path'] = record_audio()
        
        if 'recording_path' in st.session_state and st.session_state['recording_path']:
            st.audio(st.session_state['recording_path'])
            
            if st.button("🚀 Convert & Translate", type="primary", key="record_btn"):
                process_audio(
                    st.session_state['recording_path'],
                    stt,
                    translator,
                    converter,
                    source_lang_code,
                    target_lang_code if translate_option else None,
                    source_lang_name,
                    target_lang_name if translate_option else None
                )

def process_audio(file_path, stt, translator, converter, source_lang, target_lang, 
                  source_lang_name, target_lang_name):
    """Process audio file: convert -> transcribe -> translate"""
    
    try:
        with st.spinner("🔄 Processing..."):
            # Step 1: Convert to WAV
            st.info("📝 Step 1: Converting audio format...")
            wav_path = converter.convert_to_wav(file_path)
            
            # Get audio info
            audio_info = converter.get_audio_info(wav_path)
            if audio_info:
                st.success(f"✅ Converted | Duration: {audio_info['duration']:.1f}s")
            
            # Step 2: Speech to Text
            st.info(f"🎯 Step 2: Converting speech to text ({source_lang_name})...")
            transcribed_text = stt.transcribe(wav_path, source_lang)
            
            if not transcribed_text or transcribed_text.strip() == '':
                st.error("❌ No speech detected in audio. Please try again with clear audio.")
                return
            
            st.success("✅ Transcription complete!")
            
            # Display transcribed text
            st.subheader("📝 Transcribed Text")
            st.text_area(
                f"Original Text ({source_lang_name})",
                transcribed_text,
                height=150,
                key="transcribed"
            )
            
            # Step 3: Translation (if enabled)
            translated_text = None
            if target_lang:
                st.info(f"🌍 Step 3: Translating to {target_lang_name}...")
                translated_text = translator.translate(
                    transcribed_text,
                    target_language=target_lang
                )
                st.success("✅ Translation complete!")
                
                st.subheader("🌍 Translated Text")
                st.text_area(
                    f"Translated Text ({target_lang_name})",
                    translated_text,
                    height=150,
                    key="translated"
                )
            
            # Download buttons
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download transcribed text
                st.download_button(
                    label="📥 Download Original Text",
                    data=transcribed_text,
                    file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                if translated_text:
                    # Download translated text
                    st.download_button(
                        label="📥 Download Translated Text",
                        data=translated_text,
                        file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            # Success message
            st.balloons()
            st.success("🎉 Processing complete! Your text is ready.")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()


# =====================================================
# FILE 5: requirements.txt
# =====================================================

"""
streamlit
pydub
SpeechRecognition
deep-translator
openai-whisper
sounddevice
soundfile
numpy
langdetect
"""


# =====================================================
# FILE 6: README.md
# =====================================================

"""
# 🎙️ Voice Translation Tool

A powerful voice translation tool that converts speech to text and translates it to any language.

## ✨ Features

- 📁 Upload audio/video files (MP3, MP4, WAV, M4A, OGG, FLAC, AAC)
- 🎤 Record audio directly from microphone
- 🗣️ Speech-to-text conversion using Google or Whisper
- 🌍 Translate to 30+ languages
- 💾 Download results as text files
- 🎨 Beautiful and intuitive UI

## 🚀 Installation

### 1. Install Python Dependencies

```bash
pip install streamlit pydub SpeechRecognition googletrans==4.0.0-rc1
pip install openai-whisper pyaudio sounddevice soundfile numpy
```

### 2. Install FFmpeg (Required for audio conversion)

**Windows:**
- Download from: https://ffmpeg.org/download.html
- Extract and add to PATH

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### 3. Install PyAudio (For microphone recording)

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Mac/Linux:**
```bash
pip install pyaudio
```

## 📁 Project Structure

```
voice_translation_tool/
│
├── app.py                 # Main Streamlit application
├── converter.py           # Audio format conversion module
├── speech_to_text.py      # Speech recognition module
├── translator.py          # Text translation module
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── temp/                 # Temporary files (auto-created)
```

## 🎯 Usage

### Run the Application

```bash
streamlit run app.py
```

### Using the Tool

1. **Upload File Tab:**
   - Upload audio/video file
   - Select source language
   - Enable translation (optional)
   - Select target language
   - Click "Convert & Translate"

2. **Record Audio Tab:**
   - Set recording duration
   - Click "Start Recording"
   - Speak clearly
   - Click "Convert & Translate"

3. **Download Results:**
   - Download original transcription
   - Download translated text

## 🌍 Supported Languages

### Speech Recognition:
English, Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, 
Punjabi, Spanish, French, German, Italian, Japanese, Korean, Chinese, Arabic, 
Russian, Portuguese

### Translation:
30+ languages including all Indian languages, European languages, Asian languages, 
and more!

## 🔧 Troubleshooting

### FFmpeg Not Found
- Make sure FFmpeg is installed and added to system PATH
- Restart terminal/IDE after installation

### PyAudio Installation Error
- On Windows: Use `pipwin install pyaudio`
- On Mac: `brew install portaudio` then `pip install pyaudio`
- On Linux: `sudo apt-get install python3-pyaudio`

### No Audio Detected
- Check microphone permissions
- Speak clearly and close to microphone
- Ensure audio file has clear speech

### Translation Error
- Check internet connection (required for Google Translate)
- Try again after a few seconds

## 📊 Technical Details

### Speech Recognition Engines

**Google Speech Recognition (Online):**
- Fast and accurate
- Requires internet connection
- Free tier available
- Good for real-time applications

**OpenAI Whisper (Offline):**
- Highly accurate
- Works offline
- Slower processing
- Better for noisy audio

### Audio Processing

- Automatic format conversion to WAV
- Support for compressed formats
- Preserves audio quality
- Handles both audio and video files

## 🎨 UI Features

- Clean and modern interface
- Real-time processing feedback
- Progress indicators
- Audio playback
- Download buttons
- Error handling with detailed messages

## 📝 Example Use Cases

1. **Content Creation:**
   - Transcribe podcast episodes
   - Generate video subtitles
   - Create multilingual content

2. **Business:**
   - Meeting transcriptions
   - Customer call analysis
   - International communication

3. **Education:**
   - Lecture transcriptions
   - Language learning
   - Research interviews

4. **Accessibility:**
   - Voice-to-text for hearing impaired
   - Text translation for language barriers
   - Audio documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

MIT License - Feel free to use for personal and commercial projects.

## 💡 Tips for Best Results

1. **Audio Quality:**
   - Use clear, noise-free recordings
   - Speak at normal pace
   - Avoid background music

2. **Language Selection:**
   - Choose correct source language
   - Use specific dialect codes when available

3. **File Size:**
   - Compress large files before upload
   - Split very long recordings
   - Use appropriate audio bitrate

## 🆘 Support

For issues and questions:
- Check documentation
- Review error messages
- Verify all dependencies installed
- Test with sample audio files

---

**Made with ❤️ using Streamlit, Whisper, and Google APIs**
"""

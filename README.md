# PDF Text-to-Speech Reader

A Python application that reads PDF books aloud using high-quality text-to-speech synthesis. Automatically detects and starts reading from the first chapter with multiple TTS engine options for optimal voice quality.

## Features

- **Multiple TTS Engines**: Choose from traditional engines (Festival, eSpeak NG, eSpeak) or advanced neural models
- **Neural TTS Models**: High-quality AI-powered speech synthesis with GPU acceleration support
- **Intelligent Text Chunking**: Automatic text segmentation respects model token limits for error-free processing
- **Lazy Loading**: Heavy ML libraries only loaded when neural models are selected
- **Interactive Voice Selection**: Pick your preferred TTS engine and voice at startup
- **Intelligent Chapter Detection**: Automatically finds actual chapters (not subsections) using "Chapter" keyword
- **Chapter Selection**: Choose specific chapters to read or save as audio files
- **Dual Reading Modes**: Live audio playback or save chapters to WAV files
- **Smart Audio Organization**: Saves files in `pdf-tts/<book-title>/<chapter>.wav` structure
- **Segfault Prevention**: Large chapters processed in chunks to prevent crashes
- **Smart Text Processing**: Cleans and formats text for optimal speech synthesis
- **Multiple Book Support**: Browse and select from organized book collections
- **TTS Testing Suite**: Comprehensive testing tool for all TTS engines and models
- **Cross-Platform**: Works on Linux, WSL, and Windows
- **WSL Audio Support**: Automatic audio configuration for Windows Subsystem for Linux

## Installation

### Prerequisites
- Python 3.11+
- Conda (Anaconda/Miniconda)
- Linux or Windows environment

### Setup

1. **Create and activate conda environment:**
```bash
conda create -n pdf-tts python=3.11
conda activate pdf-tts
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
conda install -c conda-forge ffmpeg libstdcxx-ng
```

**Note**: Neural TTS dependencies are included in requirements.txt and will be installed automatically.

3. **Configure PDF directory paths:**
Create a `.env` file in the project directory to specify your PDF book locations:

```bash
# Copy the example and edit with your paths
cp .env.example .env
```

Edit `.env` file with your book directory paths:
```env
# Linux/WSL base directory
LINUX_BASEDIR=/home/username/Documents/Books/

# Windows base directory  
WINDOWS_BASEDIR=C:\Users\Username\Documents\Books\
```

**Important**: The .env file is ignored by git to keep personal paths private.

4. **Install system dependencies:**

**For Linux/WSL:**
```bash
sudo apt update
sudo apt install -y pulseaudio pulseaudio-utils alsa-utils ffmpeg libstdcxx-ng festival espeak-ng espeak-ng-data festvox-kallpc16k
```

**For WSL specifically:**
Run the provided audio setup script for automatic configuration:
```bash
./setup_audio.sh
```

This script installs and configures:
- PulseAudio for WSL audio support
- Festival TTS engine (natural sounding voices)
- eSpeak NG (high quality synthetic voices)
- eSpeak (basic fallback)

## Usage

### Running the Application

```bash
conda activate pdf-tts
python pdf_reader.py
```

### TTS Testing and Validation

Test all TTS engines and GPU acceleration with the comprehensive testing suite:

```bash
# Interactive mode - menu-driven testing
python test-tts.py --interactive

# List all available TTS options  
python test-tts.py --list

# Test specific neural model
python test-tts.py --model microsoft/speecht5_tts

# Test with custom text
python test-tts.py --model facebook/mms-tts-eng --text "Custom test text"

# Run full test suite
python test-tts.py
```

**Interactive Testing Features:**
- **Test all libraries** - Tests pyttsx3, festival, espeak-ng, espeak, GPU acceleration
- **Test all neural models** - Tests HuggingFace transformer models
- **Test specific library** - Choose individual TTS engines from numbered list  
- **Test specific model** - Choose neural models with custom text input
- **GPU acceleration testing** - Validates CUDA availability and PyTorch GPU support

**Available Neural TTS Models:**
- `microsoft/speecht5_tts` - SpeechT5 (Highest quality, most natural speech with speaker embeddings)
- `facebook/mms-tts-eng` - MMS TTS English (Fast, multilingual with good quality)
- `kakao-enterprise/vits-ljs` - VITS LJSpeech (Natural sounding with advanced tokenization)

**Neural TTS Features:**
- **Smart Chunking**: Automatically splits long text into model-appropriate chunks
- **Token Limit Handling**: Prevents sequence length errors with intelligent text segmentation
- **Audio Combination**: Seamlessly combines multiple chunks into single audio output
- **Error Recovery**: Graceful fallbacks and enhanced error handling for each model type

### TTS Engine Options

When you run the application, you'll be presented with TTS engine choices:

**Traditional Engines:**
1. **pyttsx3 + espeak** - Cross-platform engine with voice selection options
2. **eSpeak NG (High Quality)** - Good quality synthetic voices, faster processing  
3. **Festival (Natural Sounding)** - Best traditional quality, most human-like voices
4. **eSpeak (Basic)** - Basic quality, reliable fallback

**Neural AI Models:**
5. **SpeechT5** - High-quality neural TTS with speaker embeddings [GPU/CPU]
6. **MMS TTS English** - Fast multilingual neural TTS [GPU/CPU]
7. **VITS LJSpeech** - Natural sounding neural speech synthesis [GPU/CPU]

### Automated Testing

For quick testing with predefined selections:
```bash
# Test with Festival TTS engine (adjust indices based on your setup)
echo -e "2\n0\n37\n1\n1" | python pdf_reader.py

# Test with eSpeak NG (adjust indices based on your setup)
echo -e "1\n0\n37\n1\n1" | python pdf_reader.py
```

**Note**: The numbers correspond to:
1. TTS engine selection (0=pyttsx3, 1=eSpeak NG, 2=Festival, 3=eSpeak, 4=SpeechT5, 5=MMS TTS, 6=VITS)
2. Folder selection (0=first folder)
3. Book selection (37=specific book index - adjust for your collection)
4. Chapter selection (1=first chapter)
5. Mode selection (1=live audio, 2=save to file)

### Workflow

1. **TTS Engine Selection**: Choose your preferred text-to-speech engine and voice
2. **Book Directory Selection**: Pick a folder containing PDF books
3. **Book Selection**: Choose a specific PDF to read
4. **Chapter Selection**: Pick a specific chapter from automatically detected chapters
5. **Reading Mode**: Choose between live audio playback or saving to WAV file
6. **Audio Processing**: Enjoy high-quality speech synthesis or organized audio files

### Audio File Organization

When saving chapters to audio files, the system creates an organized directory structure:

```
pdf-tts/
├── <Book-Title-From-PDF-Metadata>/
│   ├── 1-Chapter-1-Title.wav
│   ├── 2-Chapter-2-Title.wav
│   └── ...
└── <Another-Book>/
    ├── 1-Chapter-1-Title.wav
    └── ...
```

- **Book Title Detection**: Uses PDF metadata `/Title` field when available, falls back to filename
- **Clean Naming**: Removes special characters and normalizes spaces for filesystem compatibility
- **Chapter Organization**: Each chapter saved as individual WAV file with descriptive naming

## Dependencies

### Python Packages
- `PyPDF2`: PDF reading and processing
- `pyttsx3`: Text-to-speech synthesis interface
- `torch`: Deep learning framework for neural TTS models
- `transformers`: HuggingFace transformer models for neural TTS
- `datasets`: Speaker embeddings and model data
- `soundfile`: Audio file reading/writing for neural TTS
- `phonemizer`: Text to phoneme conversion for VITS models

### System Packages
- `pulseaudio`: Audio server for Linux/WSL
- `alsa-utils`: Audio utilities
- `ffmpeg`: Audio codec support
- `libstdcxx-ng`: C++ standard library compatibility

### TTS Engines
- `festival`: High-quality natural sounding TTS engine
- `espeak-ng`: Modern eSpeak with improved voice quality
- `espeak-ng-data`: Voice data for eSpeak NG
- `festvox-kallpc16k`: Festival voice pack for better audio quality
- `espeak`: Basic fallback TTS engine

## Audio Quality Comparison

| Engine | Quality | Speed | GPU | Notes |
|--------|---------|--------|-----|-------|
| **Neural Models** |
| SpeechT5 | ★★★★★ | ★★☆☆☆ | ✓ | Highest quality, most natural speech with smart chunking |
| MMS TTS | ★★★★☆ | ★★★★☆ | ✓ | Fast neural TTS, multilingual with robust processing |
| VITS | ★★★★☆ | ★★★☆☆ | ✓ | Natural sounding with advanced tokenization handling |
| **Traditional Engines** |
| Festival | ★★★★☆ | ★★★☆☆ | ✗ | Best traditional quality |
| eSpeak NG | ★★★☆☆ | ★★★★☆ | ✗ | Good synthetic quality |
| eSpeak (pyttsx3) | ★★★☆☆ | ★★★★★ | ✗ | Configurable voices |
| eSpeak (basic) | ★★☆☆☆ | ★★★★★ | ✗ | Basic compatibility |

## Troubleshooting

### WSL Audio Issues
If you experience audio problems in WSL, run the audio setup script:
```bash
./setup_audio.sh
```

### No TTS Engines Found
Install missing engines:
```bash
sudo apt install festival espeak-ng espeak-ng-data
```

### Poor Audio Quality
Try different TTS engines in order of preference:
1. **Neural Models** (highest quality):
   - SpeechT5 - Best overall quality and naturalness
   - MMS TTS - Fast with good quality
   - VITS - Natural sounding speech
2. **Traditional engines**:
   - Festival - Best traditional quality
   - eSpeak NG - Good synthetic quality
   - eSpeak with pyttsx3 - Configurable voices

### Environment Configuration Issues
If you see errors about missing base directory configuration:

**"Error: Base directory not configured":**
- Create a `.env` file in the project directory
- Add the appropriate variable for your operating system:
  ```env
  # For Linux/WSL users
  LINUX_BASEDIR=/path/to/your/pdf/books/
  
  # For Windows users  
  WINDOWS_BASEDIR=C:\path\to\your\pdf\books\
  ```

**"Error: Configured base directory does not exist":**
- Verify the path exists and is accessible
- Check for typos in the directory path
- Ensure you have read permissions for the directory
- Create the directory if it doesn't exist

### Neural TTS Issues
If you encounter errors with neural models:

**Token Length Errors:**
- The application automatically handles token limits with smart chunking
- If you still see token errors, try selecting shorter chapters or text sections

**CUDA/GPU Issues:**
- Neural models work on both GPU and CPU
- GPU provides faster processing but is not required
- Check GPU availability with: `python -c "import torch; print(torch.cuda.is_available())"`

**Model Loading Errors:**
- Ensure stable internet connection for initial model downloads
- Models are cached locally after first download
- Try switching to a different neural model if one fails to load

**Audio Playback Issues:**
- Ensure audio system is properly configured (run `./setup_audio.sh` for WSL)
- Try different audio players: aplay, paplay, or save to file mode

## Project Structure

```
hello_pdf/
├── pdf_reader.py          # Main PDF reader application
├── test-tts.py           # Comprehensive TTS testing suite
├── setup_audio.sh        # WSL audio configuration script
├── requirements.txt      # Python dependencies
├── claude.md            # Project instructions and session history
└── README.md           # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
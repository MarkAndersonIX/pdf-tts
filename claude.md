# PDF Text-to-Speech Reader

A Python application that reads PDF books aloud using high-quality text-to-speech synthesis with automatic chapter detection and multiple TTS engine options.

## Installation

1. Create and activate conda environment:
```bash
conda create -n pdf-tts python=3.11
conda activate pdf-tts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
conda install -c conda-forge ffmpeg libstdcxx-ng
```

3. For WSL/Linux audio support:
```bash
./setup_audio.sh
```

## Running the Application

**IMPORTANT: ALWAYS activate the pdf-tts conda environment before running any code or installing packages.**

```bash
conda activate pdf-tts
python pdf_reader.py
```

### Automated Testing
```bash
# ALWAYS run in pdf-tts environment:
source ~/.bashrc && eval "$(conda shell.bash hook)" && conda activate pdf-tts

# Test PDF reader with Festival TTS (adjust folder/book indices as needed)
echo -e "2\n0\n37\n1\n1" | python pdf_reader.py

# Test PDF reader with eSpeak NG (adjust folder/book indices as needed)
echo -e "1\n0\n37\n1\n1" | python pdf_reader.py

# Test TTS engines and GPU acceleration
python test-tts.py --interactive
```

### Environment Management
- **ALL code execution must be done in the `pdf-tts` conda environment**
- **ALL package installations must be done in the `pdf-tts` conda environment**
- **Correct activation command**: `source ~/.bashrc && eval "$(conda shell.bash hook)" && conda activate pdf-tts`
- Verify environment with `conda env list` (should show `pdf-tts` with `*`)
- Never use just `conda activate pdf-tts` - it will fail without proper initialization

### TTS Testing Suite
- **Interactive testing**: `python test-tts.py --interactive`
- **List all options**: `python test-tts.py --list`
- **Test specific model**: `python test-tts.py --model microsoft/speecht5_tts`
- **GPU acceleration testing**: Automatically validates CUDA and PyTorch GPU support

## Features

- **Multiple TTS Engines**: Festival, eSpeak NG, eSpeak, pyttsx3
- **Interactive Engine Selection**: Choose TTS engine and voice at startup
- **Intelligent Chapter Detection**: Automatically finds actual chapters containing "Chapter" keyword
- **Chapter Selection**: Choose specific chapters to read or save
- **Dual Reading Modes**: Live audio playback or save to WAV files
- **Smart Text Processing**: Chunked processing prevents segfaults with large chapters
- **Audio File Management**: Saves chapter audio in organized pdf-tts subdirectories
- **Cross-Platform Audio**: WSL audio support included
- **Book Library Management**: Browse and select from organized book collections

## Dependencies

### Python Dependencies
- PyPDF2: PDF reading and processing
- pyttsx3: Text-to-speech interface

### System Dependencies
- Multiple TTS engines (festival, espeak-ng, espeak)
- PulseAudio for WSL audio support
- ffmpeg: Audio codec support
- libstdcxx-ng: C++ standard library compatibility

## Usage Workflow

1. Select TTS engine (Festival/eSpeak NG/eSpeak/pyttsx3)
2. Choose voice (if available for selected engine)
3. Select book directory from available folders
4. Choose specific PDF from the selected directory
5. Select chapter from automatically detected chapters (containing "Chapter" keyword)
6. Choose reading mode:
   - **Live Audio**: Read aloud immediately
   - **Save to File**: Create WAV file in pdf-tts subdirectory

## Chapter Features

- **Smart Detection**: Only shows actual chapters, not subsections
- **Individual Selection**: Choose specific chapters instead of reading entire book
- **Audio File Organization**: Saved files use format: `{chapter-number}-{chapter-title}.wav`
- **Segfault Prevention**: Large chapters are processed in chunks and combined
- **Error Recovery**: Individual chunk failures don't stop entire chapter processing

## Audio Quality Comparison

| Engine | Quality | Speed | Use Case |
|--------|---------|--------|----------|
| Festival | ★★★★★ | ★★★☆☆ | Best for long reading sessions |
| eSpeak NG | ★★★★☆ | ★★★★☆ | Good balance of quality/speed |
| eSpeak (basic) | ★★☆☆☆ | ★★★★★ | Fast, basic quality |
| GTTS | ★★★★☆ | ★★☆☆☆ | Natural but requires internet |

## Session Summary (June 1, 2025)

### Major Improvements Implemented

**1. Smart Chapter Detection**
- Fixed chapter listings to only show actual chapters containing "Chapter" keyword
- Eliminated subsections and individual bookmarks from chapter list
- Reduced Linux Shell Scripting Essentials from 26+ entries to 8 actual chapters

**2. Enhanced Audio Saving with Segfault Prevention**
- Implemented chunked processing for large chapters (2000 characters per chunk)
- Added WAV file combination to create single chapter files
- Prevents segmentation faults when processing large amounts of text
- Successfully tested with 33-chunk chapter processing

**3. Organized Directory Structure**
- Added book-specific subdirectories: `pdf-tts/<book-title>/<chapter>.wav`
- Smart title detection from PDF metadata `/Title` field with filename fallback
- Clean naming conventions (removes special characters, normalizes spaces)
- Example: `pdf-tts/Linux-Shell-Scripting-Essentials/1-Chapter-1-Title.wav`

**4. Temporary File Management**
- Chunk files now saved to temporary `temp/` subdirectory during processing
- Automatic cleanup of temporary files and directory after completion
- Cleaner workspace with no leftover temporary files

**5. Fixed TTS Engine Compatibility**
- Corrected Festival TTS syntax to use `text2wave` instead of invalid `--otype` flag
- Maintained compatibility across all TTS engines (Festival, eSpeak NG, eSpeak)

### Technical Details
- **Chapter Processing**: Processes chapters in manageable 2000-character chunks
- **File Combination**: Uses Python wave module for seamless WAV concatenation
- **Error Recovery**: Individual chunk failures don't stop entire chapter processing
- **Directory Structure**: `pdf-tts/<book-title>/temp/` for processing, cleaned to `pdf-tts/<book-title>/`

### Updated Documentation
- Enhanced README.md with new features and directory structure
- Updated claude.md with comprehensive feature list
- Added workflow documentation for chapter selection and dual reading modes
- Removed pdf_summ_hf.py to focus solely on pdf_reader.py functionality
- Cleaned up dependencies and documentation references
- Updated testing examples with proper command sequence explanations

### Project Cleanup
- **Removed pdf_summ_hf.py**: Eliminated duplicate PDF reader to focus on single, robust implementation
- **Simplified Dependencies**: requirements.txt now contains only essential packages for pdf_reader.py
- **Updated Testing Commands**: Corrected automated test examples to match current interface
- **Streamlined Documentation**: All references now point to unified pdf_reader.py functionality

All changes maintain backward compatibility while significantly improving functionality and reliability.
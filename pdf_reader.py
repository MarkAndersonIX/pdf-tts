import os
import platform
import sys
import re
import subprocess
import tempfile
import PyPDF2
from dotenv import load_dotenv
try:
    import pyttsx3
except Exception as ex:
    print(f"pyttsx3 was not loaded. {ex}")
    sys.exit(1)
from enum import Enum
    
# pyttsx3 for text-to-speech synthesis


class Engine(Enum):
    PYTTSX3=1
    ESPEAK_NG=2
    FESTIVAL=3
    NEURAL_TTS=4

ENGINE = Engine.PYTTSX3

def getSubfolder(basedir) -> str:
    folders = []
    print(f"Please pick a subfolder by number or 'q' to exit, followed by Enter.\n")
    # List only top-level directories in basedir
    with os.scandir(basedir) as entries:
        for entry in entries:
            if entry.is_dir():
                folder = os.path.join(basedir, entry.name)
                folders.append(folder)
    for i, path in enumerate(folders):
        print(f"{i}.\t{path}")
    if not folders:
        return basedir
    while True:
        response = input("Selection: ")
        if response == 'q':
            exit(0)
        if response.isdigit():
            response = int(response)
            if 0 <= response < len(folders):
                print(f"Folder {folders[response]} selected.")
                return folders[response]
        else:
            print(type(response), response, len(folders))

def getBookPath(basedir) -> str:
    paths = []
    print(f"Please pick a book by number or 'q' to exit, follwed by Enter.\n")
    for root, _, files in os.walk(basedir, topdown=False):
        for idx, name in enumerate(filter(lambda x: x.endswith(".pdf"), files)):
            path = os.path.join(root, name)
            print(f"\t{idx}. {path}")
            paths.append(path)
    while True:
        response = input("Selection: ")
        if response == 'q':
            exit(0)
        if response.isdigit():
            response = int(response)
            if 0 <= response < len(paths):
                print(f"Book {paths[response]} selected.")
                return paths[response]

def check_command_line_tts():
    """Check which command-line TTS engines are available"""
    available_engines = []
    
    # Test espeak-ng
    try:
        result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            available_engines.append(('espeak-ng', 'eSpeak NG (High Quality)'))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Test festival
    try:
        result = subprocess.run(['festival', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            available_engines.append(('festival', 'Festival (Natural Sounding)'))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Test espeak (fallback)
    try:
        result = subprocess.run(['espeak', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            available_engines.append(('espeak', 'eSpeak (Basic)'))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return available_engines

def check_neural_tts_availability():
    """Check if neural TTS models are available"""
    try:
        print("Checking neural TTS availability...")
        import torch
        import transformers
        
        # Check if GPU is available
        cuda_available = torch.cuda.is_available()
        
        # Available neural models
        neural_models = {
            "microsoft/speecht5_tts": "SpeechT5 (High Quality, requires speaker embeddings)",
            "facebook/mms-tts-eng": "MMS TTS English (Fast, multilingual)",
            "kakao-enterprise/vits-ljs": "VITS LJSpeech (Natural sounding)"
        }
        
        return neural_models, cuda_available
    except ImportError:
        return {}, False

def select_voice_options():
    """Let user choose TTS engine and voice"""
    print("\n=== TTS Configuration ===")
    
    # Check both pyttsx3 drivers and command-line engines
    available_options = []
    
    # Test pyttsx3 drivers first
    pyttsx3_drivers = []
    drivers_to_test = ['espeak', 'sapi5', 'nsss']
    
    for driver in drivers_to_test:
        try:
            test_engine = pyttsx3.init(driver)
            pyttsx3_drivers.append(driver)
            test_engine.stop()
            del test_engine
        except:
            pass
    
    # Add pyttsx3 options
    for driver in pyttsx3_drivers:
        available_options.append(('pyttsx3', driver, f"pyttsx3 + {driver}"))
    
    # Add command-line TTS engines
    cmd_engines = check_command_line_tts()
    for engine, desc in cmd_engines:
        available_options.append(('cmdline', engine, desc))
    
    # Add neural TTS engines
    neural_models, gpu_available = check_neural_tts_availability()
    for model_id, desc in neural_models.items():
        gpu_note = " [GPU]" if gpu_available else " [CPU]"
        available_options.append(('neural', model_id, f"{desc}{gpu_note}"))
    
    if not available_options:
        print("No TTS engines found, using default")
        return None, None, None
    
    # Show available options
    print("Available TTS engines:")
    for i, (tts_type, engine, desc) in enumerate(available_options):
        print(f"{i}. {desc}")
    
    while True:
        try:
            choice = input(f"Select TTS engine (0-{len(available_options)-1}) or Enter for auto: ").strip()
            if choice == "":
                selected_option = None
                break
            choice = int(choice)
            if 0 <= choice < len(available_options):
                selected_option = available_options[choice]
                break
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
    
    if not selected_option:
        return None, None, None
    
    tts_type, engine, desc = selected_option
    
    # Get voices for selected engine
    if tts_type == 'pyttsx3':
        try:
            test_engine = pyttsx3.init(engine)
            voices = test_engine.getProperty('voices')
            
            if voices and len(voices) > 1:
                print(f"\nAvailable voices for {desc}:")
                
                # Show voices with better formatting
                display_voices = voices
                for i, voice in enumerate(display_voices):
                    name = getattr(voice, 'name', voice.id)
                    lang = getattr(voice, 'languages', ['Unknown'])
                    if isinstance(lang, list):
                        lang = lang[0] if lang else 'Unknown'
                    print(f"{i}. {name} ({lang})")
                
                while True:
                    try:
                        voice_choice = input(f"Select voice (0-{len(display_voices)-1}) or Enter for auto: ").strip()
                        if voice_choice == "":
                            selected_voice = None
                            break
                        voice_choice = int(voice_choice)
                        if 0 <= voice_choice < len(display_voices):
                            selected_voice = display_voices[voice_choice].id
                            break
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a number")
            else:
                selected_voice = None
            
            test_engine.stop()
            del test_engine
            
        except Exception as e:
            print(f"Error testing voices: {e}")
            selected_voice = None
    elif tts_type == 'neural':
        # Neural engines use model-specific configurations
        selected_voice = None
        print(f"Using {desc}")
    else:
        # Command-line engines use default voice
        selected_voice = None
        print(f"Using {desc} with default voice")
    
    return tts_type, engine, selected_voice

#initialize engine and return the resulting object.
def init_engine(preferred_driver=None, preferred_voice=None) -> object:
    try:
        print("Initializing TTS engine...")
        
        # Use preferred driver if specified, otherwise try defaults
        if preferred_driver:
            drivers = [preferred_driver]
        else:
            # Try different drivers in order of preference
            drivers = ['espeak-ng', 'festival', 'espeak', 'sapi5', 'nsss', 'dummy']
        
        engine = None
        
        for driver in drivers:
            try:
                engine = pyttsx3.init(driver)
                print(f"Successfully initialized with driver: {driver}")
                break
            except Exception as e:
                print(f"Failed to initialize with driver {driver}: {e}")
                continue
        
        if engine is None:
            # Try default initialization as fallback
            engine = pyttsx3.init()
            print("Using default TTS driver")
        
        # Configure speech rate for better clarity
        rate = engine.getProperty('rate')
        # Set slower rate for clarity - festival handles this better
        target_rate = 150 if 'festival' in str(engine) else max(120, rate - 80)
        engine.setProperty('rate', target_rate)
        print(f"Set speech rate to: {target_rate}")
        
        # Configure volume
        engine.setProperty('volume', 0.9)
        
        # Set voice - use preferred if specified, otherwise auto-select
        voices = engine.getProperty('voices')
        if voices:
            print(f"Available voices: {len(voices)}")
            
            if preferred_voice:
                # Use preferred voice
                voice_found = False
                for voice in voices:
                    if voice.id == preferred_voice:
                        engine.setProperty('voice', voice.id)
                        print(f"Using preferred voice: {voice.id}")
                        voice_found = True
                        break
                
                if not voice_found:
                    print(f"Preferred voice {preferred_voice} not found, using auto-selection")
                    preferred_voice = None
            
            if not preferred_voice:
                # Auto-select best voice
                voice_priorities = [
                    'english',  # Any English voice
                    'en_us',    # US English
                    'en_gb',    # British English
                    'en',       # Generic English
                    'female',   # Female voices often clearer
                    'male'      # Male voices
                ]
                
                best_voice = None
                for priority in voice_priorities:
                    matching_voices = [v for v in voices if priority.lower() in v.id.lower()]
                    if matching_voices:
                        best_voice = matching_voices[0]
                        break
                
                if best_voice:
                    engine.setProperty('voice', best_voice.id)
                    print(f"Auto-selected voice: {best_voice.id}")
                else:
                    engine.setProperty('voice', voices[0].id)
                    print(f"Using default voice: {voices[0].id}")
        
        print("TTS engine initialized successfully")
        return engine
    except Exception as ex:
        print(f"Could not initialize TTS engine: {ex}")
        exit(1)

def play_data_cmdline(data: str, engine_name: str):
    """Play text using command-line TTS engines"""
    if not data:
        return
    
    # Clean the text by removing excessive whitespace and newlines
    cleaned_text = re.sub(r'\s+', ' ', data.strip())
    
    if not cleaned_text:
        return
    
    try:
        print(f"Speaking: {cleaned_text[:50]}..." if len(cleaned_text) > 50 else f"Speaking: {cleaned_text}")
        
        # Split long text into sentences for better flow
        sentences = re.split(r'[.!?]+', cleaned_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:  # Skip very short fragments
                if engine_name == 'espeak-ng':
                    # Use espeak-ng with better quality settings
                    subprocess.run(['espeak-ng', '-s', '150', '-a', '90', sentence], 
                                 check=False, capture_output=True)
                elif engine_name == 'festival':
                    # Use festival with text input
                    process = subprocess.run(['festival', '--tts'], 
                                           input=sentence, text=True, 
                                           check=False, capture_output=True)
                elif engine_name == 'espeak':
                    # Use basic espeak
                    subprocess.run(['espeak', '-s', '150', '-a', '90', sentence], 
                                 check=False, capture_output=True)
                
                # Small pause between sentences
                import time
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Error playing speech: {e}")

def play_data_neural(data: str, model_name: str, output_to_file=False, output_file=None):
    """Play text using neural TTS models with lazy loading and chunking"""
    if not data:
        return
    
    # Clean the text
    cleaned_text = re.sub(r'\s+', ' ', data.strip())
    if not cleaned_text:
        return
    
    # Define model-specific token limits (confirmed from model configs)
    model_limits = {
        "microsoft/speecht5_tts": 300,  # Actual limit is 600, use 300 for very safe chunking
        "facebook/mms-tts-eng": 800,   # Conservative estimate for MMS 
        "kakao-enterprise/vits-ljs": 600  # Conservative estimate for VITS
    }
    
    # Get token limit for chunking
    token_limit = model_limits.get(model_name, 300)
    
    # Estimate characters per token (very conservative: 2-3 chars per token)
    char_limit = token_limit * 2
    
    # Split text into chunks if needed
    if len(cleaned_text) <= char_limit:
        text_chunks = [cleaned_text]
    else:
        print(f"Text too long ({len(cleaned_text)} chars, ~{len(cleaned_text)//4} tokens), splitting into chunks for {model_name}...")
        text_chunks = []
        
        # Split by sentences and other natural breaks to maintain readability
        sentences = re.split(r'[.!?;:]+\s+', cleaned_text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed limit, save current chunk
            if len(current_chunk) + len(sentence) + 1 > char_limit and current_chunk:
                text_chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            text_chunks.append(current_chunk.strip())
        
        print(f"Split into {len(text_chunks)} chunks (max {char_limit} chars each)")
    
    try:
        print("Loading neural TTS libraries...")
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModel
        import numpy as np
        import tempfile
        import subprocess
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print(f"Processing {len(text_chunks)} chunk(s) with {model_name}...")
        
        # Process each chunk and collect audio
        all_audio = []
        sample_rate = None
        
        for i, chunk_text in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}..." if len(chunk_text) > 50 else f"Processing chunk {i+1}/{len(text_chunks)}: {chunk_text}")
            
            if model_name == "microsoft/speecht5_tts":
                if i == 0:  # Only load model components once
                    print("Loading SpeechT5 components...")
                    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                    from datasets import load_dataset
                    
                    processor = SpeechT5Processor.from_pretrained(model_name)
                    model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
                    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
                    
                    # Load speaker embeddings
                    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
                
                # Generate speech for this chunk
                inputs = processor(text=chunk_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                
                chunk_audio = speech.cpu().numpy().astype(np.float32)
                all_audio.append(chunk_audio)
                sample_rate = 16000
            
            else:
                # Use pipeline for other models
                if i == 0:  # Only load pipeline once
                    # Special handling for VITS model
                    if "vits" in model_name.lower():
                        try:
                            # Try with specific VITS pipeline parameters
                            tts_pipeline = pipeline(
                                "text-to-speech", 
                                model=model_name, 
                                device=0 if device == "cuda" else -1,
                                tokenizer_kwargs={"padding": True, "truncation": True, "max_length": 512}
                            )
                        except Exception as e:
                            print(f"VITS pipeline creation failed: {e}")
                            print("Attempting direct model loading for VITS...")
                            # Fallback to direct model loading
                            from transformers import VitsModel, VitsTokenizer
                            tokenizer = VitsTokenizer.from_pretrained(model_name)
                            model = VitsModel.from_pretrained(model_name).to(device)
                            use_direct_vits = True
                    else:
                        tts_pipeline = pipeline("text-to-speech", model=model_name, device=0 if device == "cuda" else -1)
                        use_direct_vits = False
                
                if "vits" in model_name.lower() and 'use_direct_vits' in locals() and use_direct_vits:
                    # Direct VITS model inference
                    try:
                        # Tokenize with proper padding and truncation
                        inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                        
                        # Generate speech
                        with torch.no_grad():
                            output = model(**inputs)
                            chunk_audio = output.waveform.squeeze().cpu().numpy().astype(np.float32)
                        
                        all_audio.append(chunk_audio)
                        sample_rate = model.config.sampling_rate
                    except Exception as e:
                        print(f"Direct VITS inference failed: {e}")
                        # Skip this chunk
                        continue
                else:
                    # Standard pipeline inference
                    result = tts_pipeline(chunk_text)
                    
                    chunk_audio = result["audio"]
                    if isinstance(chunk_audio, list):
                        chunk_audio = np.array(chunk_audio)
                    if chunk_audio.ndim > 1:
                        chunk_audio = chunk_audio.squeeze()
                    chunk_audio = chunk_audio.astype(np.float32)
                    
                    all_audio.append(chunk_audio)
                    sample_rate = result["sampling_rate"]
        
        # Combine all audio chunks
        if len(all_audio) == 1:
            audio_data = all_audio[0]
        else:
            print(f"Combining {len(all_audio)} audio chunks...")
            audio_data = np.concatenate(all_audio, axis=0)
        
        # Save and play audio
        import soundfile as sf
        
        if output_to_file and output_file:
            # Save to specified file
            sf.write(output_file, audio_data, samplerate=sample_rate)
            print(f"Audio saved to: {output_file}")
        else:
            # Play immediately using temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, samplerate=sample_rate)
                
                # Try to play audio using available system players
                try:
                    subprocess.run(["aplay", temp_file.name], check=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        subprocess.run(["paplay", temp_file.name], check=True, capture_output=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print(f"Could not play audio automatically. File available at: {temp_file.name}")
                        return temp_file.name  # Return temp file path if playback fails
                
                # Clean up temp file
                os.unlink(temp_file.name)
        
    except Exception as e:
        print(f"Error with neural TTS: {e}")
        return None

def play_data(data: str, engine, tts_type='pyttsx3', engine_name='espeak'):
    """Universal play function that handles pyttsx3, command-line, and neural TTS"""
    if tts_type == 'cmdline':
        play_data_cmdline(data, engine_name)
    elif tts_type == 'neural':
        play_data_neural(data, engine_name)
    else:
        # Original pyttsx3 implementation
        if not data:
            return
        
        # Clean the text by removing excessive whitespace and newlines
        cleaned_text = re.sub(r'\s+', ' ', data.strip())
        
        if not cleaned_text:
            return
        
        try:
            print(f"Speaking: {cleaned_text[:50]}..." if len(cleaned_text) > 50 else f"Speaking: {cleaned_text}")
            
            # Split long text into sentences for better flow
            sentences = re.split(r'[.!?]+', cleaned_text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    engine.say(sentence)
                    engine.runAndWait()
                    # Small pause between sentences for better clarity
                    import time
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error playing speech: {e}")

def get_chapters(pdf_reader) -> list:
    """Extract chapters from PDF bookmarks - only items containing 'chapter'"""
    chapters = []
    
    try:
        outlines = pdf_reader.outline
        if outlines:
            chapter_num = 1
            for item in outlines:
                if isinstance(item, list):
                    # Handle nested bookmarks
                    for subitem in item:
                        if hasattr(subitem, 'title') and hasattr(subitem, 'page'):
                            title = subitem.title
                            page_num = pdf_reader.get_destination_page_number(subitem)
                            if 'chapter' in title.lower():
                                chapters.append({
                                    'number': chapter_num,
                                    'title': title,
                                    'start_page': page_num
                                })
                                chapter_num += 1
                else:
                    # Handle top-level bookmarks
                    if hasattr(item, 'title') and hasattr(item, 'page'):
                        title = item.title
                        page_num = pdf_reader.get_destination_page_number(item)
                        if 'chapter' in title.lower():
                            chapters.append({
                                'number': chapter_num,
                                'title': title,
                                'start_page': page_num
                            })
                            chapter_num += 1
    except Exception as e:
        print(f"Could not read PDF bookmarks: {e}")
    
    # If no chapters found via bookmarks, create default chapters based on page ranges
    if not chapters:
        num_pages = len(pdf_reader.pages)
        pages_per_chapter = max(10, num_pages // 10)  # At least 10 pages per chapter
        chapter_num = 1
        for start_page in range(0, num_pages, pages_per_chapter):
            chapters.append({
                'number': chapter_num,
                'title': f"Chapter {chapter_num}",
                'start_page': start_page
            })
            chapter_num += 1
    
    return chapters

def find_first_chapter(pdf_reader) -> int:
    """Find the page number where the first chapter starts"""
    chapters = get_chapters(pdf_reader)
    if chapters:
        return chapters[0]['start_page']
    return 0

def select_chapter_and_mode(pdf_reader):
    """Let user select chapter and reading mode"""
    chapters = get_chapters(pdf_reader)
    
    print("\n=== Chapter Selection ===")
    print("Available chapters:")
    for chapter in chapters:  # Show all chapters
        print(f"{chapter['number']}. {chapter['title']} (Page {chapter['start_page'] + 1})")
    
    # Chapter selection
    while True:
        try:
            chapter_choice = input(f"Select chapter (1-{len(chapters)}) or Enter for auto-start: ").strip()
            if chapter_choice == "":
                selected_chapter = chapters[0] if chapters else {'number': 1, 'title': 'Chapter 1', 'start_page': 0}
                break
            chapter_choice = int(chapter_choice)
            if 1 <= chapter_choice <= len(chapters):
                selected_chapter = chapters[chapter_choice - 1]
                break
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
    
    # Mode selection
    print(f"\nSelected: {selected_chapter['title']}")
    print("Choose mode:")
    print("1. Read aloud (live audio)")
    print("2. Save to audio file")
    
    while True:
        try:
            mode_choice = input("Select mode (1-2): ").strip()
            if mode_choice in ["1", "2"]:
                save_to_file = mode_choice == "2"
                break
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter 1 or 2")
    
    return selected_chapter, save_to_file

def create_audio_filename(bookpath: str, chapter_num: int, chapter_title: str, engine_name: str, pdf_reader=None) -> str:
    """Create filename for saving audio files with book title subdirectory"""
    # Get directory containing the PDF
    pdf_dir = os.path.dirname(bookpath)
    
    # Try to get book title from PDF metadata
    book_title = None
    if pdf_reader and hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
        if '/Title' in pdf_reader.metadata:
            potential_title = pdf_reader.metadata['/Title']
            if potential_title and len(potential_title.strip()) > 0 and len(potential_title.strip()) < 100:
                book_title = potential_title.strip()
    
    # Fall back to PDF filename if no title found
    if not book_title:
        book_filename = os.path.basename(bookpath)
        book_title = os.path.splitext(book_filename)[0]
    
    # Clean book title for directory name
    clean_book_title = re.sub(r'[^\w\s-]', '', book_title)
    clean_book_title = re.sub(r'\s+', '-', clean_book_title.strip())
    
    # Create pdf-tts/book-title subdirectory
    audio_dir = os.path.join(pdf_dir, "pdf-tts", clean_book_title)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Clean chapter title for filename
    clean_title = re.sub(r'[^\w\s-]', '', chapter_title)
    clean_title = re.sub(r'\s+', '-', clean_title.strip())
    
    # Choose file extension based on engine
    if engine_name in ['espeak-ng', 'espeak', 'festival']:
        ext = 'wav'
    else:
        ext = 'wav'  # Default to WAV
    
    # Create filename: chapter-number-chapter-name.ext
    filename = f"{chapter_num}-{clean_title}.{ext}"
    
    return os.path.join(audio_dir, filename)

def save_audio_cmdline(text: str, engine_name: str, output_file: str):
    """Save text to audio file using command-line TTS engines"""
    if not text:
        return
    
    # Clean the text
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    if not cleaned_text:
        return
    
    try:
        if engine_name == 'espeak-ng':
            # Save to WAV file
            subprocess.run(['espeak-ng', '-s', '150', '-a', '90', '-w', output_file, cleaned_text], 
                         check=True)
        elif engine_name == 'festival':
            # Use festival to save to WAV with proper syntax
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_text:
                temp_text.write(cleaned_text)
                temp_text_path = temp_text.name
            
            try:
                subprocess.run(['text2wave', temp_text_path, '-o', output_file], 
                              check=True)
            finally:
                os.unlink(temp_text_path)
        elif engine_name == 'espeak':
            # Save to WAV file
            subprocess.run(['espeak', '-s', '150', '-a', '90', '-w', output_file, cleaned_text], 
                         check=True)
        
        print(f"Saved audio: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error saving audio: {e}")
    except Exception as e:
        print(f"Error saving audio: {e}")

def save_audio_pyttsx3(text: str, engine, output_file: str):
    """Save text to audio file using pyttsx3"""
    if not text:
        return
    
    try:
        # Clean the text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        if not cleaned_text:
            return
        
        # Configure engine to save to file
        engine.save_to_file(cleaned_text, output_file)
        engine.runAndWait()
        
        print(f"Saved audio: {output_file}")
        
    except Exception as e:
        print(f"Error saving audio with pyttsx3: {e}")

def play_or_save_data(data: str, engine, tts_type='pyttsx3', engine_name='espeak', 
                      save_to_file=False, output_file=None):
    """Universal function that either plays audio or saves to file"""
    if save_to_file and output_file:
        # Save to file
        if tts_type == 'cmdline':
            save_audio_cmdline(data, engine_name, output_file)
        elif tts_type == 'neural':
            play_data_neural(data, engine_name, output_to_file=True, output_file=output_file)
        else:
            save_audio_pyttsx3(data, engine, output_file)
    else:
        # Play live audio
        play_data(data, engine, tts_type, engine_name)

# Load environment variables
load_dotenv()

system = platform.system()     
if system == 'Linux':
    basedir = os.getenv('LINUX_BASEDIR')
elif system == 'Windows':
    basedir = os.getenv('WINDOWS_BASEDIR')
else:
    print(f'Unsupported OS {system}')
    exit(1)

# Validate basedir configuration
if not basedir:
    print(f"Error: Base directory not configured for {system}")
    print("Please configure the appropriate directory in your .env file:")
    if system == 'Linux':
        print("  LINUX_BASEDIR=/path/to/your/pdf/books/")
    elif system == 'Windows':
        print("  WINDOWS_BASEDIR=C:\\path\\to\\your\\pdf\\books\\")
    print("\nExample .env file content:")
    print("  # Linux/WSL base directory")
    print("  LINUX_BASEDIR=/home/user/Documents/Books/")
    print("  # Windows base directory")
    print("  WINDOWS_BASEDIR=C:\\Users\\Username\\Documents\\Books\\")
    exit(1)

if not os.path.exists(basedir):
    print(f"Error: Configured base directory does not exist: {basedir}")
    print("Please ensure the directory exists or update your .env file with the correct path.")
    exit(1)

# Let user configure TTS options
tts_type, engine_name, preferred_voice = select_voice_options()

# Initialize engine based on type
if tts_type == 'cmdline':
    engine = None  # Command-line engines don't need pyttsx3 engine
    print(f"Using command-line TTS: {engine_name}")
else:
    engine = init_engine(engine_name, preferred_voice)

subfolder = getSubfolder(basedir)
bookpath = getBookPath(subfolder)
print(f"Loading book {bookpath}...")
pdf_reader = PyPDF2.PdfReader(bookpath)
metadata = pdf_reader.metadata
num_pages = len(pdf_reader.pages)
print(f"metadata: {metadata}, num_pages: {num_pages}")

# Let user select chapter and mode
selected_chapter, save_to_file = select_chapter_and_mode(pdf_reader)

# Determine chapter end page
chapters = get_chapters(pdf_reader)
chapter_index = selected_chapter['number'] - 1
start_page = selected_chapter['start_page']

if chapter_index + 1 < len(chapters):
    end_page = chapters[chapter_index + 1]['start_page']
else:
    end_page = num_pages

print(f"\nProcessing: {selected_chapter['title']}")
print(f"Pages: {start_page + 1} to {end_page}")

if save_to_file:
    # Create audio filename
    audio_file = create_audio_filename(bookpath, selected_chapter['number'], 
                                     selected_chapter['title'], engine_name, pdf_reader)
    print(f"Saving audio to: {audio_file}")
    
    # Collect all text from the chapter first
    chapter_text = ""
    for num in range(start_page, end_page):
        if num < num_pages:
            page = pdf_reader.pages[num]
            page_text = page.extract_text()
            if page_text:
                chapter_text += page_text + " "
    
    # Process chapter in chunks to avoid segfaults
    if chapter_text.strip():
        # Split into manageable chunks (by sentences)
        sentences = re.split(r'[.!?]+', chapter_text)
        chunks = []
        current_chunk = ""
        max_chunk_size = 2000  # Characters per chunk
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 2 < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Create temporary directory for chunk files
        temp_dir = os.path.join(os.path.dirname(audio_file), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save chunks as temporary files, then combine
        temp_files = []
        print(f"Processing chapter in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            temp_file = os.path.join(temp_dir, f'chunk_{i}.wav')
            temp_files.append(temp_file)
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            try:
                play_or_save_data(chunk, engine, tts_type, engine_name, 
                                 save_to_file=True, output_file=temp_file)
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                continue
        
        # Try to combine files if multiple chunks exist
        if len(temp_files) > 1:
            try:
                # Simple concatenation for WAV files
                import wave
                with wave.open(audio_file, 'wb') as output:
                    for i, temp_file in enumerate(temp_files):
                        if os.path.exists(temp_file):
                            try:
                                with wave.open(temp_file, 'rb') as input_wav:
                                    if i == 0:
                                        output.setparams(input_wav.getparams())
                                    output.writeframes(input_wav.readframes(input_wav.getnframes()))
                            except Exception as e:
                                print(f"Error reading {temp_file}: {e}")
                
                # Clean up temp files and directory
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
                        
            except Exception as e:
                print(f"Error combining files: {e}")
                # If combining fails, just keep the first chunk as the output
                if temp_files and os.path.exists(temp_files[0]):
                    import shutil
                    shutil.move(temp_files[0], audio_file)
                    # Clean up remaining temp files and directory
                    for temp_file in temp_files[1:]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
        elif temp_files and os.path.exists(temp_files[0]):
            # Single chunk, just rename it and clean up temp directory
            import shutil
            shutil.move(temp_files[0], audio_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    print(f"Chapter audio saved successfully!")
else:
    # Live reading mode
    print('Playing PDF File..')
    for num in range(start_page, end_page):
        if num < num_pages:
            page = pdf_reader.pages[num]
            data = page.extract_text()
            if data:
                play_or_save_data(data, engine, tts_type, engine_name, 
                                 save_to_file=False)

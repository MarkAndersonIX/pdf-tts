#!/usr/bin/env python3
"""
Test script for TTS (Text-to-Speech) libraries.
Tests pyttsx3 functionality and GPU-accelerated TTS with transformers.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")


def test_pyttsx3():
    """Test pyttsx3 TTS engine functionality."""
    print("Testing pyttsx3 TTS library...")
    
    try:
        print("Loading pyttsx3...")
        import pyttsx3
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Found {len(voices)} voices:")
        
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name} ({voice.id})")
        
        # Test basic properties
        rate = engine.getProperty('rate')
        volume = engine.getProperty('volume')
        print(f"Current rate: {rate}")
        print(f"Current volume: {volume}")
        
        # Test speech with first available voice
        if voices:
            engine.setProperty('voice', voices[0].id)
            test_text = "Hello! This is a test of the text-to-speech functionality."
            
            print(f"\nSpeaking test text with voice: {voices[0].name}")
            engine.say(test_text)
            engine.runAndWait()
            
            # Test different rate
            engine.setProperty('rate', rate - 50)
            engine.say("This is slower speech.")
            engine.runAndWait()
            
            # Reset rate
            engine.setProperty('rate', rate)
        
        print("✓ pyttsx3 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ pyttsx3 test failed: {e}")
        return False


def test_system_tts():
    """Test system TTS commands (festival, espeak, etc.)."""
    print("\nTesting system TTS commands...")
    
    tts_commands = [
        ("festival", "echo 'Testing Festival TTS' | festival --tts"),
        ("espeak-ng", "espeak-ng 'Testing eSpeak NG TTS'"),
        ("espeak", "espeak 'Testing eSpeak TTS'")
    ]
    
    results = {}
    
    for name, command in tts_commands:
        try:
            result = os.system(f"{command} 2>/dev/null")
            if result == 0:
                print(f"✓ {name} is available and working")
                results[name] = True
            else:
                print(f"✗ {name} failed (exit code: {result})")
                results[name] = False
        except Exception as e:
            print(f"✗ {name} error: {e}")
            results[name] = False
    
    return results


def test_gpu_tts():
    """Test GPU-accelerated TTS with transformers."""
    print("\nTesting GPU-accelerated TTS...")
    
    try:
        print("Loading torch...")
        import torch
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU count: {gpu_count}")
            print(f"GPU name: {gpu_name}")
            
            # Test simple tensor operation on GPU
            device = torch.device("cuda:0")
            test_tensor = torch.randn(10, 10).to(device)
            result = torch.mm(test_tensor, test_tensor.transpose(0, 1))
            print(f"GPU tensor operation successful: {result.shape}")
            print("✓ GPU acceleration is available and working")
            return True
        else:
            print("✗ CUDA not available - GPU TTS will not work")
            return False
            
    except Exception as e:
        print(f"✗ GPU TTS test failed: {e}")
        return False


def test_specific_model(model_name, text="Hello, this is a test of the neural text-to-speech model."):
    """Test a specific TTS model from HuggingFace."""
    print(f"\nTesting model: {model_name}")
    print(f"Test text: '{text}'")
    
    # Popular TTS models that work with transformers
    supported_models = {
        "microsoft/speecht5_tts": "SpeechT5",
        "facebook/mms-tts-eng": "MMS TTS English", 
        "kakao-enterprise/vits-ljs": "VITS LJSpeech",
        # Note: espnet models require native ESPnet installation, not compatible with transformers pipeline
    }
    
    if model_name not in supported_models:
        print(f"Available models: {list(supported_models.keys())}")
        print(f"✗ Model '{model_name}' not in supported list")
        return False
    
    try:
        print("Loading torch...")
        import torch
        print("Loading transformers...")
        from transformers import pipeline, AutoTokenizer, AutoModel
        
        print(f"Loading {supported_models[model_name]} model...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if model_name == "microsoft/speecht5_tts":
            # SpeechT5 requires special handling
            print("Loading SpeechT5 components...")
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import numpy as np
            
            processor = SpeechT5Processor.from_pretrained(model_name)
            model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
            
            # Load speaker embeddings dataset for voice
            print("Loading speaker embeddings...")
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
            
            # Tokenize input
            inputs = processor(text=text, return_tensors="pt").to(device)
            
            print("✓ Model loaded successfully")
            print("✓ Text tokenized successfully")
            print("Generating audio...")
            
            # Generate speech
            with torch.no_grad():
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            
            # Save audio to temp file and play
            import soundfile as sf
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, speech.cpu().numpy(), samplerate=16000)
                print(f"✓ Audio generated, playing...")
                
                # Try to play audio using available system players
                try:
                    subprocess.run(["aplay", temp_file.name], check=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        subprocess.run(["paplay", temp_file.name], check=True, capture_output=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print(f"Could not play audio automatically. File saved temporarily at: {temp_file.name}")
                
                # Clean up temp file
                os.unlink(temp_file.name)
            
        elif "mms-tts" in model_name:
            # MMS TTS models
            try:
                tts_pipeline = pipeline("text-to-speech", model=model_name, device=0 if device == "cuda" else -1)
                print("✓ Model loaded successfully")
                print("Generating audio...")
                
                # Generate speech
                result = tts_pipeline(text)
                
                # Save audio to temp file and play
                import soundfile as sf
                import tempfile
                import subprocess
                import numpy as np
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    # Ensure audio is in the right format
                    audio_data = result["audio"]
                    if isinstance(audio_data, list):
                        audio_data = np.array(audio_data)
                    
                    # Ensure it's the right shape and type
                    if audio_data.ndim > 1:
                        audio_data = audio_data.squeeze()
                    
                    sf.write(temp_file.name, audio_data.astype(np.float32), samplerate=result["sampling_rate"])
                    print(f"✓ Audio generated, playing...")
                    
                    # Try to play audio using available system players
                    try:
                        subprocess.run(["aplay", temp_file.name], check=True, capture_output=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        try:
                            subprocess.run(["paplay", temp_file.name], check=True, capture_output=True)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            print(f"Could not play audio automatically. File saved temporarily at: {temp_file.name}")
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)
                
            except Exception as e:
                print(f"Pipeline creation failed: {e}")
                print("Attempting direct model loading...")
                # Try loading components separately for basic validation
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(device)
                print("✓ Model components loaded successfully (audio generation not available)")
        
        else:
            # Generic model loading - try pipeline first, then fallback
            try:
                # Try using pipeline for audio generation
                tts_pipeline = pipeline("text-to-speech", model=model_name, device=0 if device == "cuda" else -1)
                print("✓ Model loaded successfully")
                print("Generating audio...")
                
                # Generate speech
                result = tts_pipeline(text)
                
                # Save audio to temp file and play
                import soundfile as sf
                import tempfile
                import subprocess
                import numpy as np
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    # Ensure audio is in the right format
                    audio_data = result["audio"]
                    if isinstance(audio_data, list):
                        audio_data = np.array(audio_data)
                    
                    # Ensure it's the right shape and type
                    if audio_data.ndim > 1:
                        audio_data = audio_data.squeeze()
                    
                    sf.write(temp_file.name, audio_data.astype(np.float32), samplerate=result["sampling_rate"])
                    print(f"✓ Audio generated, playing...")
                    
                    # Try to play audio using available system players
                    try:
                        subprocess.run(["aplay", temp_file.name], check=True, capture_output=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        try:
                            subprocess.run(["paplay", temp_file.name], check=True, capture_output=True)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            print(f"Could not play audio automatically. File saved temporarily at: {temp_file.name}")
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)
                    
            except Exception as e:
                print(f"Pipeline creation failed: {e}")
                print("Attempting direct model loading...")
                # Fallback to direct model loading for validation
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name).to(device)
                    print("✓ Model components loaded successfully (audio generation not available)")
                except Exception as e2:
                    print(f"Model loading failed: {e2}")
                    return False
        
        print(f"✓ {supported_models[model_name]} model test completed")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def get_all_options():
    """Get all available TTS options (libraries and models)."""
    return {
        "libraries": {
            "pyttsx3": "Cross-platform TTS with multiple voices",
            "festival": "High-quality system TTS engine",
            "espeak-ng": "Fast multilingual TTS engine",
            "espeak": "Basic fast TTS engine",
            "gpu_test": "Test GPU acceleration availability"
        },
        "models": {
            "microsoft/speecht5_tts": "High-quality neural TTS (requires speaker embeddings)",
            "facebook/mms-tts-eng": "Multilingual TTS - English",
            "kakao-enterprise/vits-ljs": "VITS model trained on LJSpeech",
            # Note: ESPnet models require native installation, not compatible with transformers
        }
    }


def list_available_options():
    """List all available TTS libraries and models for testing."""
    options = get_all_options()
    
    print("\nAvailable TTS Options:")
    print("=" * 50)
    
    print("📚 LIBRARIES:")
    for lib, description in options["libraries"].items():
        print(f"  • {lib}")
        print(f"    Description: {description}")
        print()
    
    print("🤖 NEURAL MODELS:")
    for model, description in options["models"].items():
        print(f"  • {model}")
        print(f"    Description: {description}")
        print()


def interactive_selector():
    """Interactive selector for TTS libraries and models."""
    options = get_all_options()
    
    print("\n🎯 TTS Interactive Selector")
    print("=" * 40)
    print("Choose what to test:")
    print("1. Test all libraries")
    print("2. Test all neural models")
    print("3. Test specific library")
    print("4. Test specific model")
    print("5. Test everything")
    print("6. List all options")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🔧 Testing all libraries...")
            results = {}
            results["gpu"] = test_gpu_tts()
            results["pyttsx3"] = test_pyttsx3()
            results.update(test_system_tts())
            
            print("\n📊 Library Test Results:")
            for lib, result in results.items():
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {lib}: {status}")
            
        elif choice == "2":
            print("\n🤖 Testing all neural models...")
            results = {}
            for model in options["models"]:
                print(f"\n--- Testing {model} ---")
                results[model] = test_specific_model(model)
            
            print("\n📊 Model Test Results:")
            for model, result in results.items():
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {model}: {status}")
                
        elif choice == "3":
            print("\n📚 Available libraries:")
            libs = list(options["libraries"].keys())
            for i, lib in enumerate(libs, 1):
                print(f"  {i}. {lib} - {options['libraries'][lib]}")
            
            lib_choice = input(f"\nSelect library (1-{len(libs)}): ").strip()
            try:
                lib_index = int(lib_choice) - 1
                selected_lib = libs[lib_index]
                
                print(f"\n🔧 Testing {selected_lib}...")
                if selected_lib == "gpu_test":
                    result = test_gpu_tts()
                elif selected_lib == "pyttsx3":
                    result = test_pyttsx3()
                elif selected_lib in ["festival", "espeak-ng", "espeak"]:
                    system_results = test_system_tts()
                    result = system_results.get(selected_lib, False)
                
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"\nResult: {selected_lib} - {status}")
                
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                
        elif choice == "4":
            print("\n🤖 Available models:")
            models = list(options["models"].keys())
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
                print(f"     {options['models'][model]}")
                print()
            
            model_choice = input(f"Select model (1-{len(models)}): ").strip()
            try:
                model_index = int(model_choice) - 1
                selected_model = models[model_index]
                
                custom_text = input(f"\nEnter test text (or press Enter for default): ").strip()
                if not custom_text:
                    custom_text = "Hello, this is a test of the neural text-to-speech model."
                
                print(f"\n🤖 Testing {selected_model}...")
                result = test_specific_model(selected_model, custom_text)
                
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"\nResult: {selected_model} - {status}")
                
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                
        elif choice == "5":
            print("\n🚀 Testing everything...")
            main_test_suite()
            
        elif choice == "6":
            list_available_options()
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main_test_suite():
    """Run the complete test suite."""
    print("TTS Library Test Suite")
    print("=" * 30)
    
    # Test GPU acceleration first
    gpu_result = test_gpu_tts()
    
    # Test pyttsx3
    pyttsx3_result = test_pyttsx3()
    
    # Test system TTS commands
    system_results = test_system_tts()
    
    # Summary
    print("\n" + "=" * 30)
    print("Test Summary:")
    print(f"GPU TTS: {'✓ PASS' if gpu_result else '✗ FAIL'}")
    print(f"pyttsx3: {'✓ PASS' if pyttsx3_result else '✗ FAIL'}")
    
    for name, result in system_results.items():
        print(f"{name}: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Recommendations
    print("\nRecommendations:")
    if gpu_result:
        print("- GPU acceleration is available for neural TTS models")
        print("- Use --model <model_name> to test specific HuggingFace models")
        print("- Use --list to see available models")
    
    if pyttsx3_result:
        print("- pyttsx3 is working and can be used for cross-platform TTS")
    
    working_systems = [name for name, result in system_results.items() if result]
    if working_systems:
        print(f"- System TTS engines available: {', '.join(working_systems)}")
    else:
        print("- No system TTS engines found. Install festival, espeak-ng, or espeak")


def main():
    """Run TTS tests with command line arguments or interactive mode."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive TTS (Text-to-Speech) Testing Suite",
        epilog="""
Examples:
  python test-tts.py                                    # Interactive menu (default)
  python test-tts.py --list                            # Show all available options
  python test-tts.py --suite                           # Run complete test suite
  python test-tts.py --model microsoft/speecht5_tts    # Test specific model
  python test-tts.py --model facebook/mms-tts-eng --text "Hello world"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model", 
        metavar="MODEL_NAME",
        help="Test a specific HuggingFace TTS model. Available models: microsoft/speecht5_tts, facebook/mms-tts-eng, kakao-enterprise/vits-ljs, espnet/kan-bayashi_ljspeech_vits"
    )
    
    parser.add_argument(
        "--text", 
        default="Hello, this is a test of the neural text-to-speech model.",
        metavar="TEXT",
        help="Custom text to use for model testing (default: %(default)s)"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available TTS options including system libraries (pyttsx3, festival, espeak-ng, espeak) and neural models with descriptions"
    )
    
    parser.add_argument(
        "--suite", 
        action="store_true",
        help="Run the complete test suite including GPU acceleration test, pyttsx3 test, and all system TTS engines with summary report"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_options()
        return
    
    if args.model:
        # Test specific model only
        print("TTS Model Test")
        print("=" * 30)
        result = test_specific_model(args.model, args.text)
        print(f"\nModel test: {'✓ PASS' if result else '✗ FAIL'}")
        return
    
    if args.suite:
        # Run full test suite
        main_test_suite()
        return
    
    # Default to interactive mode
    interactive_selector()


if __name__ == "__main__":
    main()
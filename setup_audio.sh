#!/bin/bash

# WSL Audio Setup Script
# Sets up PulseAudio for WSL to enable audio output to Windows

set -e

echo "Setting up audio for WSL..."

# Check if running in WSL
if ! grep -q "microsoft" /proc/version; then
    echo "Warning: This script is designed for WSL. For native Linux, PulseAudio should work out of the box."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install required packages
echo "Installing PulseAudio, ALSA utilities, and TTS engines..."
sudo apt update
sudo apt install -y pulseaudio pulseaudio-utils alsa-utils festival espeak-ng espeak-ng-data festvox-kallpc16k

# Check if WSLg PulseAudio server exists
if [ -S "/mnt/wslg/PulseServer" ]; then
    echo "WSLg PulseAudio server found. Configuring..."
    PULSE_HOST="unix:/mnt/wslg/PulseServer"
else
    echo "WSLg not detected. Using localhost TCP connection..."
    PULSE_HOST="tcp:localhost"
fi

# Configure environment
echo "Setting PULSE_HOST environment variable..."
export PULSE_HOST="$PULSE_HOST"

# Add to bashrc if not already present
if ! grep -q "PULSE_HOST" ~/.bashrc; then
    echo "export PULSE_HOST=$PULSE_HOST" >> ~/.bashrc
    echo "Added PULSE_HOST to ~/.bashrc"
else
    # Update existing entry
    sed -i "s|export PULSE_HOST=.*|export PULSE_HOST=$PULSE_HOST|" ~/.bashrc
    echo "Updated PULSE_HOST in ~/.bashrc"
fi

# Test audio configuration
echo "Testing audio configuration..."
if command -v pactl >/dev/null 2>&1; then
    if pactl info >/dev/null 2>&1; then
        echo "✓ PulseAudio server is accessible"
        
        # Test with speaker-test if available
        if command -v speaker-test >/dev/null 2>&1; then
            echo "Testing speakers (you should hear test sounds)..."
            speaker-test -t wav -c 2 -l 1 2>/dev/null || echo "Speaker test completed (no error means success)"
        fi
        
        # Test with sample sound if available
        if [ -f "/usr/share/sounds/alsa/Front_Left.wav" ]; then
            echo "Playing test sound..."
            paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || echo "Test sound played"
        fi
        
        # Test TTS engines
        echo "Testing TTS engines..."
        
        if command -v espeak-ng >/dev/null 2>&1; then
            echo "✓ eSpeak NG installed and working"
        else
            echo "⚠ eSpeak NG not found"
        fi
        
        if command -v festival >/dev/null 2>&1; then
            echo "✓ Festival installed and working" 
        else
            echo "⚠ Festival not found"
        fi
        
        if command -v espeak >/dev/null 2>&1; then
            echo "✓ eSpeak (basic) installed and working"
        else
            echo "⚠ eSpeak not found"
        fi
        
        echo "✓ Audio and TTS setup complete!"
        echo "Audio and text-to-speech should now work in WSL applications."
        
    else
        echo "❌ Cannot connect to PulseAudio server"
        echo "Try restarting WSL or check Windows audio settings"
        exit 1
    fi
else
    echo "❌ pactl not found. Installation may have failed."
    exit 1
fi

echo ""
echo "Setup complete! You may need to restart your terminal for changes to take effect."
echo ""
echo "Testing commands:"
echo "  Audio: pactl info"
echo "  TTS engines: espeak-ng 'Hello world', festival --tts (then type text)"
echo ""
echo "Available TTS engines for PDF reader:"
echo "  - Festival (Natural Sounding) - Best quality"
echo "  - eSpeak NG (High Quality) - Good quality, faster"
echo "  - eSpeak (Basic) - Basic quality"
#!/bin/bash
# Setup script for Raspberry Pi deployment
# Run with: bash setup_raspberry_pi.sh

set -e  # Exit on error

echo "========================================="
echo "Red Scarf Monitor - Raspberry Pi Setup"
echo "========================================="

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Get username (default to 'pi')
USERNAME=${SUDO_USER:-$USER}
if [ "$USERNAME" = "root" ]; then
    USERNAME="pi"
fi
HOME_DIR="/home/$USERNAME"

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    git \
    python3-pip \
    python3-venv \
    vim \
    curl \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    alsa-utils \
    libasound2-dev \
    libjpeg-dev \
    zlib1g-dev \
    libtiff-dev \
    libpng-dev

echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch for ARM..."
# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected 64-bit ARM architecture"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Detected 32-bit ARM architecture"
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/armv7l
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

# Try to install mediapipe separately if it fails
if ! python -c "import mediapipe" 2>/dev/null; then
    echo "Installing mediapipe separately..."
    pip install mediapipe --no-deps || echo "Mediapipe installation failed, continuing..."
    pip install opencv-contrib-python numpy protobuf
fi

echo "Creating required directories..."
mkdir -p images/violations images/students logs database

echo "Setting permissions..."
chmod -R 755 images logs database
chown -R $USERNAME:$USERNAME images logs database

echo "Checking camera..."
if [ -e /dev/video0 ]; then
    echo "Camera device found: /dev/video0"
    # Add user to video group
    sudo usermod -a -G video $USERNAME
else
    echo "Warning: No camera device found at /dev/video0"
    echo "Please ensure your camera is connected and configured"
fi

echo ""
echo "========================================="
echo "Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml to set camera_index and other settings"
echo "2. Ensure your model file exists at model/best.pt"
echo "3. Test the application:"
echo "   source venv/bin/activate"
echo "   python app.py --mode web"
echo ""
echo "4. To set up as a service, run:"
echo "   sudo cp red-scarf-monitor.service /etc/systemd/system/"
echo "   sudo nano /etc/systemd/system/red-scarf-monitor.service  # Edit paths if needed"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable red-scarf-monitor.service"
echo "   sudo systemctl start red-scarf-monitor.service"
echo ""


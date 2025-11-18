# Deployment Guide: Raspberry Pi

This guide will help you deploy the Red Scarf Monitoring application to a Raspberry Pi running Linux.

## Prerequisites

- Raspberry Pi 4 (recommended) or Raspberry Pi 3B+
- Raspberry Pi OS (64-bit recommended for better PyTorch support)
- MicroSD card (32GB+ recommended)
- USB camera or Raspberry Pi Camera Module
- Internet connection for initial setup

## Step 1: Initial Raspberry Pi Setup

### 1.1 Install Raspberry Pi OS

1. Download Raspberry Pi OS (64-bit) from [raspberrypi.org](https://www.raspberrypi.org/software/)
2. Flash the image to your MicroSD card using Raspberry Pi Imager
3. Enable SSH during initial setup (or create `ssh` file in boot partition)
4. Boot your Raspberry Pi

### 1.2 Initial System Updates

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv vim curl
```

### 1.3 Configure Camera

**For USB Camera:**
```bash
# Test camera
lsusb  # Should show your camera
```

**For Raspberry Pi Camera Module:**
```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
# Reboot after enabling
sudo reboot

# Test camera
libcamera-hello --list-cameras
```

## Step 2: Transfer Application Files

### Option A: Using Git (Recommended)

```bash
cd ~
git clone <your-repository-url> check_khanquang
cd check_khanquang
```

### Option B: Using SCP (from your development machine)

```bash
# On your development machine
scp -r /path/to/check_khanquang pi@<raspberry-pi-ip>:~/
```

### Option C: Using USB Drive

1. Copy files to USB drive
2. Mount on Raspberry Pi:
```bash
sudo mkdir /mnt/usb
sudo mount /dev/sda1 /mnt/usb  # Adjust device name as needed
cp -r /mnt/usb/check_khanquang ~/
```

## Step 3: Python Environment Setup

### 3.1 Create Virtual Environment

```bash
cd ~/check_khanquang
python3 -m venv venv
source venv/bin/activate
```

### 3.2 Install System Dependencies

```bash
# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv libgl1-mesa-glx libglib2.0-0

# Install audio dependencies (for alerts)
sudo apt install -y alsa-utils libasound2-dev

# Install other dependencies
sudo apt install -y libjpeg-dev zlib1g-dev libtiff-dev libpng-dev
```

### 3.3 Install Python Dependencies

**Important:** PyTorch for ARM requires special installation:

```bash
# Install PyTorch for ARM64 (Raspberry Pi OS 64-bit)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For 32-bit OS, use:
# pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/armv7l

# Install other requirements
pip install -r requirements.txt

# If mediapipe fails, try:
pip install mediapipe --no-deps
pip install opencv-contrib-python numpy protobuf
```

**Note:** PyTorch installation may take 30-60 minutes on Raspberry Pi.

## Step 4: Configuration

### 4.1 Update Camera Index

Edit `config/config.yaml`:

```yaml
camera_index: 0  # Usually 0 for USB camera, or use camera path for Pi Camera
```

**For Raspberry Pi Camera Module, you may need to use:**
```yaml
camera_index: "/dev/video0"  # or the appropriate device
```

### 4.2 Verify Model File

Ensure your YOLO model file exists:
```bash
ls -lh model/best.pt
# If missing, copy it or download a pre-trained model
```

### 4.3 Create Required Directories

```bash
mkdir -p images/violations images/students logs database
```

### 4.4 Test Camera Access

```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera opened:', cap.isOpened()); cap.release()"
```

## Step 5: Create Systemd Service (Auto-start on Boot)

### 5.1 Create Service File

```bash
sudo nano /etc/systemd/system/red-scarf-monitor.service
```

Add the following content:

```ini
[Unit]
Description=Red Scarf Monitoring Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/check_khanquang
Environment="PATH=/home/pi/check_khanquang/venv/bin"
ExecStart=/home/pi/check_khanquang/venv/bin/python /home/pi/check_khanquang/app.py --mode web
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Note:** Replace `pi` with your username if different.

### 5.2 Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable red-scarf-monitor.service

# Start service now
sudo systemctl start red-scarf-monitor.service

# Check status
sudo systemctl status red-scarf-monitor.service

# View logs
sudo journalctl -u red-scarf-monitor.service -f
```

## Step 6: Network Configuration

### 6.1 Find Raspberry Pi IP Address

```bash
hostname -I
# or
ip addr show
```

### 6.2 Access Web Interface

From another device on the same network:
```
http://<raspberry-pi-ip>:8000
```

### 6.3 Configure Static IP (Optional)

Edit network configuration:
```bash
sudo nano /etc/dhcpcd.conf
```

Add (adjust for your network):
```
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

## Step 7: Performance Optimization

### 7.1 Increase GPU Memory Split (for Pi Camera)

```bash
sudo raspi-config
# Advanced Options > Memory Split > 128 (or higher)
```

### 7.2 Enable GPU Acceleration (Optional)

For better OpenCV performance:
```bash
sudo apt install -y libgles2-mesa-dev
```

### 7.3 Optimize Model Size

Consider using a smaller YOLO model:
- `yolov8n.pt` (nano) - fastest, less accurate
- `yolov8s.pt` (small) - balanced
- `yolov8m.pt` (medium) - better accuracy, slower

Update `config/config.yaml`:
```yaml
model_path: yolov8n.pt  # Use nano model for better performance
```

### 7.4 Reduce Image Size

In `config/config.yaml`:
```yaml
img_size: 416  # Reduce from 640 for faster processing
```

## Step 8: Security Considerations

### 8.1 Change Default Password

```bash
passwd
```

### 8.2 Configure Firewall (UFW)

```bash
sudo apt install ufw
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # Web interface
sudo ufw enable
```

### 8.3 Update Admin Password

Edit `config/config.yaml`:
```yaml
admin_password: 'your-secure-password-here'
```

### 8.4 Use HTTPS (Optional - for production)

Consider using a reverse proxy with nginx and Let's Encrypt:
```bash
sudo apt install nginx certbot python3-certbot-nginx
```

## Step 9: Monitoring and Maintenance

### 9.1 Check Service Status

```bash
sudo systemctl status red-scarf-monitor.service
```

### 9.2 View Logs

```bash
# Real-time logs
sudo journalctl -u red-scarf-monitor.service -f

# Last 100 lines
sudo journalctl -u red-scarf-monitor.service -n 100
```

### 9.3 Restart Service

```bash
sudo systemctl restart red-scarf-monitor.service
```

### 9.4 Check Disk Space

```bash
df -h
# Monitor images/violations directory size
du -sh images/violations
```

### 9.5 Set Up Log Rotation

Create log rotation config:
```bash
sudo nano /etc/logrotate.d/red-scarf-monitor
```

Add:
```
/home/pi/check_khanquang/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

## Step 10: Troubleshooting

### Camera Not Detected

```bash
# List video devices
ls -l /dev/video*

# Test with v4l2
v4l2-ctl --list-devices

# Check permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### Application Won't Start

1. Check Python version:
```bash
python3 --version  # Should be 3.8+
```

2. Verify virtual environment:
```bash
source venv/bin/activate
which python
```

3. Test imports:
```bash
python3 -c "import cv2, torch, ultralytics; print('OK')"
```

### Low Performance

1. Check CPU temperature:
```bash
vcgencmd measure_temp
```

2. Consider adding a heat sink or fan
3. Reduce model size or image resolution
4. Close unnecessary services

### Port Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill <PID>
```

Or change port in `app.py` (search for port 8000).

## Step 11: Backup Strategy

### 11.1 Backup Database

```bash
# Create backup script
nano ~/backup.sh
```

Add:
```bash
#!/bin/bash
BACKUP_DIR=~/backups
mkdir -p $BACKUP_DIR
cp ~/check_khanquang/database/violations.db $BACKUP_DIR/violations_$(date +%Y%m%d).db
# Keep only last 7 days
find $BACKUP_DIR -name "violations_*.db" -mtime +7 -delete
```

Make executable:
```bash
chmod +x ~/backup.sh
```

### 11.2 Schedule Automatic Backups

```bash
crontab -e
```

Add:
```
0 2 * * * /home/pi/backup.sh
```

## Quick Start Checklist

- [ ] Raspberry Pi OS installed and updated
- [ ] Camera configured and tested
- [ ] Application files transferred
- [ ] Python virtual environment created
- [ ] Dependencies installed (including PyTorch)
- [ ] Configuration file updated (camera index, model path)
- [ ] Required directories created
- [ ] Systemd service created and enabled
- [ ] Service started and running
- [ ] Web interface accessible
- [ ] Firewall configured
- [ ] Backup strategy in place

## Additional Resources

- [Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)
- [PyTorch ARM Installation](https://pytorch.org/get-started/locally/)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)

## Support

If you encounter issues:
1. Check service logs: `sudo journalctl -u red-scarf-monitor.service -n 50`
2. Verify camera access: `ls -l /dev/video*`
3. Test Python environment: `source venv/bin/activate && python app.py --mode web`
4. Check system resources: `htop` or `free -h`


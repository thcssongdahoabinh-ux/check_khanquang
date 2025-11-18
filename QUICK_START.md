# Quick Start Guide - Raspberry Pi Deployment

## Fast Setup (5-10 minutes)

### 1. Initial Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install basic tools
sudo apt install -y git python3-pip python3-venv
```

### 2. Get Application
```bash
cd ~
# Option A: Clone from git
git clone <your-repo-url> check_khanquang
cd check_khanquang

# Option B: Transfer files via SCP or USB
```

### 3. Run Setup Script
```bash
chmod +x setup_raspberry_pi.sh
bash setup_raspberry_pi.sh
```

**Note:** PyTorch installation may take 30-60 minutes. Be patient!

### 4. Configure
```bash
# Edit configuration
nano config/config.yaml

# Important settings to check:
# - camera_index: 0 (or your camera device)
# - model_path: model/best.pt (ensure file exists)
```

### 5. Test Run
```bash
source venv/bin/activate
python app.py --mode web
```

Open browser: `http://<raspberry-pi-ip>:8000`

### 6. Install as Service (Auto-start)
```bash
# Copy service file (edit paths if needed)
sudo cp red-scarf-monitor.service /etc/systemd/system/

# Edit if your username/path is different
sudo nano /etc/systemd/system/red-scarf-monitor.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable red-scarf-monitor.service
sudo systemctl start red-scarf-monitor.service

# Check status
sudo systemctl status red-scarf-monitor.service
```

## Common Issues

### Camera Not Working
```bash
# Check camera
ls -l /dev/video*
v4l2-ctl --list-devices

# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

### PyTorch Installation Fails
```bash
# For 64-bit OS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For 32-bit OS
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/armv7l
```

### Service Won't Start
```bash
# Check logs
sudo journalctl -u red-scarf-monitor.service -n 50

# Verify paths in service file
sudo nano /etc/systemd/system/red-scarf-monitor.service
```

### Low Performance
- Use smaller model: `yolov8n.pt` instead of `yolov8l.pt`
- Reduce image size in config: `img_size: 416`
- Ensure adequate cooling (heat sink/fan)

## Find Your Raspberry Pi IP
```bash
hostname -I
```

## Access Web Interface
From any device on the same network:
```
http://<raspberry-pi-ip>:8000
```

## Useful Commands

```bash
# View service logs
sudo journalctl -u red-scarf-monitor.service -f

# Restart service
sudo systemctl restart red-scarf-monitor.service

# Stop service
sudo systemctl stop red-scarf-monitor.service

# Check service status
sudo systemctl status red-scarf-monitor.service
```

For detailed information, see `DEPLOYMENT_RASPBERRY_PI.md`


# LersGAN
**LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement**

## 📈 Outputs & Applications
<img src="photo/demo1.png" width="75%" />

## 🎨 Visual Comparisons on RSDark Dataset
<img src="photo/RSph.png" width="75%" />

## 📰 News
- **2025-04-15** — 🚀 Released code and pretrained models for LersGAN.
- **2025-04-14** — 🔧 Initialized the Git project.

## 📝 TODO
- [ ] Update training scripts
- [ ] Update testing scripts
- [ ] Integrate baseline methods
- [ ] Improve Linux visualization scripts

## 🚀 Getting Started

### 📦 Environment Setup
```bash
# Requires Python 3.5+
pip install -r requirements.txt
```
- **Hardware**: At least 3× NVIDIA 1080 Ti GPUs (or adjust batch size accordingly).

### 📂 Directory Structure
```
.
├── photo/           # Sample images and demos
├── CEAM/            # Code of CEAM
├── LersGAN-main/    # Main code of LersGAN
│   └── model/       # Pretrained models
│       └── VGG/     # Download VGG pretrained model here
└── requirements.txt
```

### 🔗 Download Pretrained VGG Model
1. Download from [Google Drive](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing)
2. Place the file in `model/VGG/`.

## 🤖 Training
TODO
<!-- > **Note**: Start a Visdom server for real-time monitoring:
```bash
nohup python -m visdom.server --port=8097 &
```
Then run:
```bash
python scripts/train.py
``` -->

## 🧪 Testing
TODO
<!-- 1. Create test directories:
   ```bash
   mkdir -p test_dataset/testA test_dataset/testB
   ```
2. Place your low-light images in `test_dataset/testA` (and keep one image in `test_dataset/testB`).
3. Download pretrained model and place it in `checkpoints/enlightening/`.
4. Run:
   ```bash
   python scripts/test.py
   ``` -->

## 📫 Contact
For issues or contributions, please open an issue or pull request on GitHub.

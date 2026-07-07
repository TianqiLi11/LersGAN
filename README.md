# LersGAN

Reproduction code and dataset tooling for **LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement**.

- Paper: https://doi.org/10.1109/JSTARS.2025.3608696
- Dataset: https://huggingface.co/datasets/cod-tdq/lersgan

## Outputs & Applications

<img src="photo/demo1.png" width="75%" />

## Visual Comparisons on RSDark Dataset

<img src="photo/RSph.png" width="75%" />

## Installation

```bash
git clone https://github.com/TianqiLi11/LersGAN.git
cd LersGAN
pip install -r requirements.txt
```

## Dataset

The released dataset follows the paper's `B. Dataset Description` structure. It contains unpaired low-light and normal-light training/validation images, plus paired remote-sensing test sets for `RSDark1` and `RSDark2`.

```bash
huggingface-cli download cod-tdq/lersgan \
  --repo-type dataset \
  --local-dir data/LERSGAN_RS_PAPER
```

Expected layout:

```text
data/LERSGAN_RS_PAPER/
├── train/
│   ├── low_light/
│   └── normal_light/
├── val/
│   ├── low_light/
│   └── normal_light/
├── test/
│   ├── normal/
│   ├── RSDark1/
│   └── RSDark2/
└── README.md
```

Counts:

- Train: 800 low-light images and 900 normal-light images.
- Validation: 100 low-light images and 100 normal-light images.
- Test: 500 normal-light remote-sensing images, 500 RSDark1 images, and 500 RSDark2 images.

## Training

The default configuration trains on the unpaired paper dataset at `data/LERSGAN_RS_PAPER`.

```bash
python train.py --config configs/lersgan_default.yaml
```

Useful overrides:

```bash
python train.py \
  --config configs/lersgan_default.yaml \
  --data-root data/LERSGAN_RS_PAPER \
  --output-dir experiments/lersgan_rs \
  --batch-size 8 \
  --epochs 100
```

Training outputs are written to:

```text
experiments/lersgan_rs/
├── checkpoints/
├── samples/
├── logs/
└── config.yaml
```

## Testing

Run inference on the configured test domains:

```bash
python test.py \
  --config configs/lersgan_default.yaml \
  --checkpoint experiments/lersgan_rs/checkpoints/latest.pt \
  --split test
```

By default, the script evaluates both `RSDark1` and `RSDark2`. Enhanced images are saved under `results/lersgan_rs`; PSNR/SSIM are computed when matching normal-light target images are present.

To test one domain:

```bash
python test.py \
  --config configs/lersgan_default.yaml \
  --checkpoint experiments/lersgan_rs/checkpoints/latest.pt \
  --input-domain RSDark1
```

## Rebuilding The Dataset

The released Hugging Face dataset is the recommended path. The local rebuild scripts are kept for reproducibility and record source selections in manifests.

```bash
python scripts/build_paper_dataset.py --source-config configs/paper_dataset_sources.json --check-only
bash scripts/run_build_paper_dataset.sh
```

The paper reports aggregate selected counts, not per-source selected counts. This rebuild therefore records the actual selected local files in `manifests/*.csv`.

## Citation

```bibtex
@article{li2025lersgan,
  title = {LersGAN: A GAN-Based Model for Low-Light Remote Sensing Image Enhancement},
  author = {Li, TianQi},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year = {2025},
  doi = {10.1109/JSTARS.2025.3608696},
  publisher = {IEEE}
}
```

# [SARA: Structure-Aware Riemannian-Guided Alignment](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/8852.pdf)



## Repository Structure

```text
SARA/
├── train.py                     # Distributed training entry point
├── test_single.py               # Single-run retrieval evaluation
├── data.py                      # Dataset and DataLoader definitions
├── engine.py                    # Training and validation loops
├── utils.py                     # Losses, metrics, and checkpoint utilities
├── vocab.py                     # Vocabulary loading utilities
├── layers/                      # Model and feature-extraction modules
│   ├── VCSR.py                  # Main retrieval model
│   ├── spdnet.py                # SPD feature processing
│   ├── lesa_modules.py          # Log-Euclidean structural alignment
│   ├── rest_v2.py               # RestV2 visual encoder
│   ├── xvlm.py                  # Vision-language model components
│   └── ...                      # Supporting layers and operations
├── option/                      # Dataset and experiment YAML files
├── vocab/                       # Dataset vocabulary JSON files
├── bert-base-uncased/           # BERT tokenizer/config files
├── Rct/                         # Retrieval-related utility scripts
└── util/                        # Data conversion utilities
```

The following are created or prepared locally and are not tracked by Git:

- `data/`: datasets and preprocessed features
- `restv2_base_224.pth` and `bert-base-uncased/pytorch_model.bin`: pretrained weights
- `checkpoint/`: saved model checkpoints
- `logs/`: TensorBoard and training logs

Datasets, pretrained weights, checkpoints, and TensorBoard logs are not tracked by Git.

## Environment

* Python 3.8.0
* Pytorch 1.11.0
* torchvision 0.12.0
* librosa 0.9.1
* numpy 1.21.6
* tqdm 4.64.0

The training script requires Python, PyTorch with CUDA support, and the packages imported by this project, including `torchvision`, `transformers`, `geoopt`, `timm`, `PyYAML`, `Pillow`, `nltk`, `numpy`, `pandas`, `tensorboard_logger`, `tqdm`, `click`, `scikit-learn`, and `scipy`. Install a PyTorch/torchvision pair compatible with your CUDA installation, then install the remaining packages in your environment.

Training uses `torch.distributed` with the NCCL backend and starts one worker per visible CUDA GPU. Run commands from the repository root.

## Data and Pretrained Weights

Download and prepare the dataset files separately. Dataset paths are configured in `option/ERA_VCSR.yaml` and `option/UDV_VCSR.yaml`; the default configurations expect data under `data/`.

Place the RestV2 checkpoint at `restv2_base_224.pth`. Place the BERT-base-uncased tokenizer and model files under `bert-base-uncased/` (`vocab.txt`, tokenizer/config files, and `pytorch_model.bin`). These large files are intentionally excluded from Git. The BERT files must be available locally; the code does not download them at runtime.

## Training

Train with the ERA configuration:

```bash
python train.py --path_opt option/ERA_VCSR.yaml
```

Train with the UDV configuration:

```bash
python train.py --path_opt option/UDV_VCSR.yaml
```

## Evaluation

Pass the configuration and the checkpoint to evaluate:

```bash
python test_single.py \
  --path_opt option/ERA_VCSR.yaml \
  --resume checkpoint/path/to/best.pth.tar
```

The default checkpoint path in `test_single.py` is specific to the authors' local experiment and may not exist in a fresh checkout. Always pass `--resume` explicitly.

## Citation

```bibtex
@inproceedings{lu2026sara,
  title     = {SARA: Structure-Aware Riemannian-Guided Alignment for Drone Image-Text Retrieval},
  author    = {Lu, Keyu and Ma, Qing and Lu, Zhenyu and Bai, Cong},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  publisher = {Springer Nature}
}
```

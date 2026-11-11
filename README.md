# SARA: Structure-Aware Riemannian-Guided Alignment

Code for drone image-text retrieval with semantic representations and SPD-based structural alignment.

## Repository Structure

- `train.py`: distributed training entry point
- `test_single.py`: single-run retrieval evaluation
- `engine.py`, `data.py`, `utils.py`: training, data loading, and shared utilities
- `layers/`: visual/text encoders, VCSR model, SPD modules, and alignment losses
- `option/`: YAML experiment configurations
- `vocab/`: dataset vocabulary files

Datasets, pretrained weights, checkpoints, and TensorBoard logs are not tracked by Git.

## Environment

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

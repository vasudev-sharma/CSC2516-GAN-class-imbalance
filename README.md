# Computer Vision Project: GAN-based Data Augmentation for Chest X-ray Classification

- Cloned from [this repository](https://github.com/ssundaram21/6.819FinalProjectRAMP)
- Paper: https://arxiv.org/pdf/2107.02970.pdf

## Download Data
```bash
bash bash_scripts/run_data.sh
```

## Baseline (Supervised DenseNet121)
```python
python main.py --with_gan "" --idx 0 --user "vasu" --skip_training ""  --dataset_size  10  --dataset  "COVID"  --fraction  0.5  --epochs  30  --data_aug  "True"
```
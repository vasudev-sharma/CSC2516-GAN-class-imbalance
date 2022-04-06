# Computer Vision Project: GAN-based Data Augmentation for Chest X-ray Classification

- Cloned from [this repository](https://github.com/ssundaram21/6.819FinalProjectRAMP)
- Paper: https://arxiv.org/pdf/2107.02970.pdf

## Download Data
```bash
$ bash bash_scripts/run_data.sh
```
| Dataset | Link | Download Directory
| -----   | :----:| ----: |
|   RSNA      |  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data    | `data/RSNA_Pneumonia`
|   COVID-19      |  https://github.com/muhammedtalo/COVID-19    | `data/COVID-19`
|   COVID-chestxray-dataset      |    https://github.com/ieee8023/covid-chestxray-dataset  | `data/covid-chestxray-dataset`

## Baseline (Supervised DenseNet121)
```python
$ python main.py --with_gan "" --idx 0 --user "vasu" --skip_training ""  --dataset_size  10  --dataset  "COVID"  --fraction  0.5  --epochs  30  --data_aug  "True"
```

For more info, read about the arguments `$ python main.py --help`
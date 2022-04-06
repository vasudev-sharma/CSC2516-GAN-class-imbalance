# Computer Vision Project: GAN-based Data Augmentation for Chest X-ray Classification



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


## GANs: TODO
| **GAN** | **Path** | **Loss** |
| :---- | :-----: | -----: |
|DCGAN|`scripts/DCGAN.py`| BCE Loss|
|DCGAN with GP|`scripts/DCGAN_GP.py`| W-Loss + Gradient Penalty|
| Conditional DCGAN with GP | `scripts/DCGAN_GP_conditional.py`| W-Loss + Gradient Penalty|
| SNGAN | `scripts/SNGAN_conditional.py` | Spectral Normalization | 

## Acknowledgements: TODO
This repository makes use of the code from the following repositories. We thank all the authors for making their code publically available.
- [this repository](https://github.com/ssundaram21/6.819FinalProjectRAMP)


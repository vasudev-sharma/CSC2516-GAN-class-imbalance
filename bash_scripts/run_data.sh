#! /bin/bash

cd data/covid-chestxray-dataset
git clone https://github.com/ieee8023/covid-chestxray-dataset.git 
mv covid-chestxray-dataset/* . & rm covid-chestxray-dataset
cd ..

# Download RSNA dataset: 
curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10338/862042/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1649475858&Signature=fRPS%2FtsckbbbEUQszmuLB3P8faPu1nWNL7q6fXanSiVfNEtHhFeapc3GIDpgvJhDhtGeCWUqMzGWTLTXpMnnTpwCKBq0eVM87hvlrOUlfk%2FNXAPVQ64AoWHQ%2B0Rb38iomWBJU2G7RBRtk5AkCCX3RViSOF47BY2bvjZskmdupRj%2B1I0UWjdGYKCRlFt3V7POvb2ojEXVaLp9lIEcaLA7PiJVQV%2B8QFRNdK6sQtqCDjxU6PC9e58MsqyuMDTbn3HoF0k7p2SkbgWxTbxdUTn%2FXQdKgrmCNqTkAAviZk66xd8q4bkTAUvvVKaH0gmSM2rs2UgOm3xrGo5Uhsnz23vitg%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-pneumonia-detection-challenge.zip' \
  -H 'authority: storage.googleapis.com' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-user: ?1' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'referer: https://www.kaggle.com/' \
  -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' \
  --compressed -o data.zip
unzip data.zip -d RSNA_Pneumonia 
rm data.zip

git clone https://github.com/muhammedtalo/COVID-19.git

cd ..

# Install python dependecies
pip install torchxrayvision
pip install pydicom
pip install wandb


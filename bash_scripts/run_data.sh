#! /bin/bash

cd data/covid-chestxray-dataset
git clone https://github.com/ieee8023/covid-chestxray-dataset.git 
mv covid-chestxray-dataset/* . & rm covid-chestxray-dataset
cd ..

# Download RSNA dataset: 
curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10338/862042/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1648620756&Signature=r48LMathzZauJ1hRKjcwAU71s6%2B4OC7iE5zMVLd3QePhfCNQDfga0wt8P6dgYffuzdhGA%2Fa%2Bc%2Fc1zNnO76N%2F5Rk17HMQyeWB4yrHpexcdODTAbtv2cvpUr%2BqeUHjpL2o2WWBDA8K2L644XJi8tT2TgNzswJZI1AU5UJMik5FwuYUsjJBJeTgBEjNcw7MHxHuE3Kvq%2FPggZkS8RnluHgE%2B8jmIrwGPKODzXidNtMFbVt5GCV2W4Da6eFhDl2ciTjaeplJ7qSz4pkbiWjgzBw7yry%2B6wxjsJbiO6RsKkIb84DgONUO2JjdQaj%2FFMwf4lu9QUqXc6%2FJCmwd2RovUgY%2B0g%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-pneumonia-detection-challenge.zip' \
  -H 'authority: storage.googleapis.com' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36' \
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

cd ..

# Install python dependecies
pip install torchxrayvision
pip install pydicom

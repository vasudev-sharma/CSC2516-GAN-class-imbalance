#! /bin/bash

cd data/covid-chestxray-dataset
git clone https://github.com/ieee8023/covid-chestxray-dataset.git 
<<<<<<< HEAD
mv covid-chestxray-dataset/* . & rm -r covid-chestxray-dataset
cd ..


cd CSC2547/datasets/RSNA_Pneumonia
curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10338/862042/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1648272700&Signature=Jz7aHzeVnZoKQ95OhiV8JRs3Bjk5frJFObxxigpjM3aa0oxUJcwkDnWc3AJQ5eoAAjn0DAJhoNiUZPhxnzkyVTY7qNoi0nXavORbnqz4pRVC8Ptb09IY%2FCNXt1c%2FlHMGJY8ODC6t6G5AaYSAC1%2BE93AJPODw9kkjajQpl1QXxGsZ6iCsmO0TA22Zxb6CwjaoiXVQ1xuiJ55E9Ukbg2dAK6QQCS1gi6cMHpqFyIp%2BvNDO1Aj3b%2Fqwzl3YCEYq%2FxMqJW%2BQcjNZ%2F0ond2uQMl6wagICb9Gdx3kB47BWOm5wsj13Q1NjAJSL77W2aHUx4jzNTBZdX3ceAwQL1ok%2FDStZWw%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-pneumonia-detection-challenge.zip' \
=======
mv covid-chestxray-dataset/* . & rm covid-chestxray-dataset
cd ..

# Download RSNA dataset: 
curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10338/862042/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1649794531&Signature=D6ySHCsl5wJFuZ%2F9Y%2Fr86PxHpj4v%2BiUHWX9946NzY9vEQnfrl8bIwgKp5nAvHpv7ith3ZfvGSNwjp08ftX2u0VZoLIDLUCvLT5PN6U7omPlrKw2YIU%2FUUGvTN2wGYoJSep0eoABmLfBRZUkVdmOi1LDK%2FtptrFf8opkHBL8gt0tufp2GaUJ15Iq83Jor8QM0CPDF8Oj4OwjnE%2B2GPNWDPlBWMzsXlfR%2FWfTOAD9Xjj1qMCErXr%2FnMBwMhGAP1jmMMJOnHQcyPIGf9TZWCKxM14ZvCTiZOZ53ScjPtaxaGHuVPpmAqmTkdD2RcyEdoqC%2FOW%2BfAg4g19cDIBzULV2uvg%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-pneumonia-detection-challenge.zip' \
>>>>>>> vasu/baseline
  -H 'authority: storage.googleapis.com' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' \
  -H 'referer: https://www.kaggle.com/' \
  -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36' \
  --compressed -o data.zip
unzip data.zip -d RSNA_Pneumonia 
rm data.zip

rm -r COVID-19
git clone https://github.com/muhammedtalo/COVID-19.git

cd ..

# Install python dependecies
pip install torchxrayvision
pip install pydicom
pip install wandb


# srun --partition biggpunodes --nodelist gpunode23 -c 2 --gres=gpu:1 --mem=31000M bash_scripts/run_train.sh

# export PYTHON to path
export PYTHONPATH="${PYTHONPATH}:/w/247/vasu/CSC2516-GAN-class-imbalance"
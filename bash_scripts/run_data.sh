#! /bin/bash

cd data/covid-chestxray-dataset
git clone https://github.com/ieee8023/covid-chestxray-dataset.git 
mv covid-chestxray-dataset/* . & rm covid-chestxray-dataset
cd ..

# Download RSNA dataset
curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10338/862042/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1648341989&Signature=MIKaXDiRHbnCPdirs%2FW7VxAY53MDxEi2NNMFB72EKMtKARq%2BZ8jNwX%2FYVR3P%2BqlWiYA7tmmrLFHvCWey2tPT6WnUGDwo6n2FbRF7hRmGEnsiiP6jKtFq%2FFBz2mdHKSVE%2FGlvaSBV7%2FufHXphfXB1Ol%2Ff75R0HcN%2BAw58otztJD5k6KazmVE63ih6BnxjEt%2F1HuZEx3ZzjHhkqBu4adqQA6J3wsLjNc8oumJUKuZUE95fKl89SeNamZFxKHzj02IRMDF1osjCrTz95xHhqlPGeHt%2BmYItcAs%2BstQA8qvl9%2BhZtBfT2cQNoB0PgItrsbdIiPEAoAtOjSIqmGSYYNooLQ%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-pneumonia-detection-challenge.zip' \
  -H 'authority: storage.googleapis.com' \
  -H 'cache-control: max-age=0' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36' \
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

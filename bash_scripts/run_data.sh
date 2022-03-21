#! /bin/bash

cd data/covid-chestxray-dataset
git clone https://github.com/ieee8023/covid-chestxray-dataset.git 
mv covid-chestxray-dataset/* . & rm covid-chestxray-dataset
cd ..


curl 'https://storage.googleapis.com/kaggle-data-sets/17810/23812/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220319%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220319T215442Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=853ab12073e1008961b421e269d76ed97545de3edd21097bc4046abb0f98cb62d166f8a1911f85a3b469433ee483ecd182eeee72312454e561bd13822cdfc32a6e2ab35b0a85f648942dfe372f63f1d0549d39e5f03ba5eb9abddb70be70e3865795ecc6fc047bf0c256712fbfc8cbce39a3b3ab6641bcc741ed4e36e5fb60650bc8e387fbe6b57546b7cc359b41c2d6e6a4b442a44ae97f4db79a2b064e16394b6d8bfca90d9814b6ec56da8d508c1e169e80da550412dd52e62c5e06c515d050f3326f4fc58dd47dbc0bcca8e5890691f72f9dfd4ff63c2373db27c01cde5e8db7ea6cebe2b58005879fb12be8b4e6bda290823402c139d919832d9d953014' \
  -H 'authority: storage.googleapis.com' \
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



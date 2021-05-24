# Datasets

- bitstamp_data.csv
  - License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
  - Owner: [Zielak](https://www.kaggle.com/mczielinski)
  - Source: https://www.kaggle.com/mczielinski/bitcoin-historical-data
  - Frequency: by minute.
- Glassnode json files
  - License: [Terms of use](https://glassnode.com/terms-and-conditions)
  - Owner: Glassnode
  - Source: https://glassnode.com/metrics under Tier 1
  - Frequency: daily 
- Alternative.me csv file
  - License: Unknown
  - Owner: alternative.me
  - Source: https://api.alternative.me/fng/?limit=3000&format=csv
  - Frequency: daily
- Google Trends:
  - License: please visit https://support.google.com/trends/answer/4365538?hl=es
  - Owner: Google
  - Frequency: weekly
  - Source: https://trends.google.com/trends/explore?date=2019-12-30%202020-12-30&q=bitcoin

To zip:

```
zip models.zip *.pickle
zip -r -s 50m datasets.zip . -x \*.pickle -x \*.zip
```

To unzip:

```
zip -F datasets.zip --out single-archive-datasets.zip
unzip single-archive-datasets.zip
unzip models.zip
```

import os
import logging
import json
import pandas as pd


def data_paths_from_periodicity(periodicity):
    if periodicity == 'hourly':
        return ['../datasets/bitstamp_data_hourly.csv']
    elif periodicity == 'daily':
        return ['../datasets/bitstamp_data_daily.csv']
    return ['../datasets/bitstamp_data.csv.part1',
            '../datasets/bitstamp_data.csv.part2',
            '../datasets/bitstamp_data.csv.part3',
            '../datasets/bitstamp_data.csv.part4',
            '../datasets/bitstamp_data.csv.part5']


def load_btc_data(periodicity):
    file_paths = data_paths_from_periodicity(periodicity)
    # Función que permite convertir el formato de las fechas como unix time
    # en un objeto de fecha.
    def unix_time_to_date(x): return pd.to_datetime(x, unit='s')
    li = []
    for filename in file_paths:
        df = pd.read_csv(filename, parse_dates=[
                         'Timestamp'], date_parser=unix_time_to_date, index_col='Timestamp')
        li.append(df)
    return pd.concat(li, axis=0)


def load_btc_csv(filepath):
    # Función que permite convertir el formato de las fechas como unix time
    # en un objeto de fecha.
    def unix_time_to_date(x): return pd.to_datetime(x, unit='s')
    return pd.read_csv(filepath, parse_dates=['Timestamp'], date_parser=unix_time_to_date, index_col='Timestamp')


def load_glassnode_json():
    glassnode_json_directory = '../datasets/glassnode/json/'

    df = pd.DataFrame()
    for f in os.listdir(glassnode_json_directory):
        if f.endswith('.json'):
            col_name = f[:-len('.json')]
            df0 = pd.read_json(os.path.join(glassnode_json_directory, f),
                               orient='records', precise_float=True,
                               convert_dates=['t'])
            # Sets the index
            df0.rename(columns={'t': 'Timestamp'}, inplace=True)
            df0.set_index('Timestamp', inplace=True)
            # Change column name
            if 'v' in df0.columns:
                df0.rename(columns={'v': col_name}, inplace=True)
            else:
                columns = df0['o'][0].keys()
                # TODO: stock-to-flow.json requires a special treatment.
                if 'ratio' in columns:
                    df0['ratio'] = df0['o'].apply(lambda x: x['ratio'])
                    df0['daysTillHalving'] = df0['o'].apply(
                        lambda x: x['daysTillHalving'])
                else:
                    for c in columns:
                        df0[[c]] = df0['o'].map(lambda d: d[c])
                df0.drop(['o'], axis=1, inplace=True)
            # Merge it
            if df.empty:
                df = df0
            else:
                df = pd.merge(df, df0, how='inner', left_index=True,
                              right_index=True)
    return df


def load_glassnode_csv():
    return load_btc_csv('../datasets/glassnode/csv/dataset.csv')


def load_gtrends_csv():
    # Correctly parses the date.
    def date_to_pandas_datetime(x): return pd.to_datetime(x, format='%Y-%m-%d')
    df = pd.read_csv('../datasets/google_trends/gtrends.csv', parse_dates=[
                     'Timestamp'], date_parser=date_to_pandas_datetime, index_col='Timestamp')
    df.sort_index(inplace=True)
    return df


def load_alternative_me_csv():
    # Correctly parses the date.
    def date_to_pandas_datetime(x): return pd.to_datetime(x, format='%d-%m-%Y')
    df = pd.read_csv('../datasets/alternative_me/alternative_me.csv', parse_dates=[
                     'Timestamp'], date_parser=date_to_pandas_datetime, index_col='Timestamp')
    # Convert SentimentClassification into a factor
    df['SentimentClassificationFactor'], _ = pd.factorize(
        df.SentimentClassification)
    # Removes the used column
    df.drop('SentimentClassification', inplace=True, axis=1)
    df.sort_index(inplace=True)
    return df

from bs4 import BeautifulSoup
from numpy import NaN
import requests

import urllib.parse

class ExtractInformation:
    def __init__(self):
        pass

    def extractData(self, df):
        df['area'] = df['area'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
        df['rooms'] = df['rooms'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
        df['bathroom'] = df['bathroom'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
        df['price'] = df['price'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)

        return df

    def getLongLat(self, df):
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(df) +'?format=json'
        response = requests.get(url).json()
    
        if response is None or len(response) == 0:
            return NaN
    
        return response[0]["lon"] +','+ response[0]["lat"]

    def filter(self, df):
        df = df.dropna()

        df.drop_duplicates(subset=['address','area','rooms','bathroom','price'], keep=False)
        df.drop(df[df.rooms > 5].index, inplace=True)
        df.drop(df[df['area'] > 120].index, inplace=True)
        df.drop(df[df['area'] < 70].index, inplace=True)

        df = df.join(df['latlon'].str.split(',', expand=True).rename(columns={0:'longitude', 1:'latitude'}))
        
        df.drop(df[df['price'] > 1000000].index, inplace=True)

        df.drop(df[df['price'] < 100000].index, inplace=True)

        df = df.drop(['address','latlon'], axis=1)

        return df
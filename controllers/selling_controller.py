from utils.DataManager import DataManager
from utils.ExtractInformation import ExtractInformation
from utils.AI import AI
from utils.PreTreatment import PreTreatment
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

import json
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import geobr

from sklearn.kernel_ridge import KernelRidge

import plotly.express as px

import numpy as np

from sklearn.linear_model import Ridge as RLR
from sklearn.metrics import r2_score
import itertools
from itertools import permutations

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.linear_model import LinearRegression as LR

class SellingController():
    def __init__(self,db):
        self.db: DataManager = db
        self._extract: ExtractInformation = ExtractInformation()
        self._gan: AI = AI()
        self._treatment: PreTreatment = PreTreatment()

    def sellingApartamentos(self):
        paginas = 200
        df = self.db.getData(paginas)

        df = self._extract.extractData(df)

        df['latlon'] = df['address'].apply(self._extract.getLongLat)

        df = self._extract.filter(df)

        df.to_csv('56pages.csv')

        visualTable = df.head(n=10).to_dict()

        return {'status': 'OK',
                'data': {
                    'visualTable': visualTable
                }
        }

    def learningApartamentos(self):
        df0 = pd.read_csv('53pages.csv')
        df1 = pd.read_csv('54pages.csv')
        df2 = pd.read_csv('55pages.csv')

        df3 = pd.read_csv('56pages.csv')
        li = []
        li.append(df0)
        li.append(df1)
        li.append(df2)
        li.append(df3)
        df = pd.concat(li, axis=0, ignore_index=True)
        
        df.drop(df[df['price'] > 1000000].index, inplace=True)

        df.drop(df[df['rooms'] > 5].index, inplace=True)
        df.drop(df[df['bathroom'] > 5].index, inplace=True)

        df.drop(df[df['longitude'] < -48.7].index, inplace=True)
        #df = df.drop(['Unnamed: 0'], axis=1)
        print(df.head())

        df['latitude2'] = df['latitude']
        df['longitude2'] = df['longitude']

        df['latitude'] = df['latitude'].round(1)
        df['longitude'] = df['longitude'].round(1)
        df.drop_duplicates(subset=['area','rooms','bathroom','longitude','latitude','price'], keep=False, inplace=True)
      
        
        #print(df.longitude.min(),df.longitude.max())

        df.longitude = (df.longitude - df.longitude.min())
        df.latitude = (df.latitude - df.latitude.min())

        #df = df.round(2)
        df['latitude'] = df['latitude'].round(2)
        df['longitude'] = df['longitude'].round(2)
        
        ylist = [a/10 for a in range(0,int(1+10*df.latitude.max()))]
        xlist = [a/10 for a in range(0,int(1+10*df.longitude.max()))]

        listaxy = []
        for combx in xlist:
            for comby in ylist:
                listaxy.append((combx,comby))

        dcord = pd.get_dummies(pd.DataFrame(listaxy, columns=['lonx','laty']), columns=['lonx','laty'])


        df = pd.get_dummies(df, columns=['latitude','longitude'], prefix=['laty','lonx'])

        X, y = self._treatment.disjoinedDf(df)

        x_trn, x_tst, y_trn, y_tst = self._treatment.disjoinedXtrain(X,y)

        '''kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel( 1e-1)
        kernel = WhiteKernel( 1e-1)

        kernel_ridge = KernelRidge(kernel=ExpSineSquared())

        
        kernel_ridge.fit(x_trn, y_trn)

        kernel_ridge.kernel
        param_distributions = {
            "alpha": loguniform(1e0, 1e3),
            "kernel__length_scale": loguniform(1e-2, 1e2),
            "kernel__periodicity": loguniform(1e0, 1e1),
        }
        kernel_ridge_tuned = RandomizedSearchCV(
            kernel_ridge,
            param_distributions=param_distributions,
            n_iter=500,
            random_state=0,
        )

        kernel_ridge_tuned.fit(x_trn, y_trn)
        
        kernel_ridge_tuned.best_params_
        df['prediction'] = kernel_ridge_tuned.predict(X) #* ( df.price.max() - df.price.min() ) + df.price.min()
        print(r2_score(y,df['prediction']))
        print(r2_score(y_tst,kernel_ridge_tuned.predict(x_tst)))'''
    
        #model, accuracy = self._gan.createModel(x_trn,x_tst,y_trn,y_tst, y.mean(), y.std())
        #X = self._treatment.expandDims(X)

        
        gaussian_process = GaussianProcessRegressor(alpha=2e-10)
        gaussian_process.fit(x_trn, y_trn)

        df['prediction_gpr'], df['std_prediction_gpr'] = gaussian_process.predict(X,return_std=True)
        print(r2_score(y,df['prediction_gpr']))
        
        df['value'] = -(df['price'] - df['prediction_gpr'])

        fig = px.scatter_mapbox(df,  lat='latitude2', lon='longitude2', color='value', hover_name=df.index, size='price',  size_max=10, zoom=1, color_continuous_scale=px.colors.diverging.RdYlGn, text='link')   

        fig.update_layout(mapbox_style="carto-positron")
        
        fig.show()
        df.to_csv('withprediction.csv')
               
        return {'status': 'OK',
                'data': {
                    'longitude': df['longitude2'].tolist(), 
                    'latitude': df['latitude2'].tolist(),
                    'price': df['price'].tolist(),
                    'link': df['link'].tolist(),
                    'value': df['value'].tolist(),
                    'rgb': df['value'].apply(lambda x: (156,204,101,0.75) if x>=0 else (239,83,80,0.75)).tolist()
                }
        }

        
from sklearn import model_selection as ms
from sklearn import preprocessing as pp
import numpy as np

class PreTreatment:
    def __init__(self):
        pass

    def disjoinedDf(self,df):
        #df = df.astype({"longitude":'float64', "latitude":'float64'})

        y = df['price']

        #df['theta'], df['radius'] = self.radialValues(df['longitude'],df['latitude'])

        df.to_csv('df_lat.csv')
        X = df.drop(['price','link','longitude2','latitude2'],axis=1)

        return X, y

    def radialValues(self,x,y):
        y += 27.59
        x += 48.48
        theta = np.arctan(y/x)
        radius = np.sqrt(x**2 + y**2)
        return theta,radius

    def disjoinedXtrain(self, X, y):
        x_trn, x_tst, y_trn, y_tst = ms.train_test_split(X,y,test_size=0.25)

        return x_trn, x_tst, y_trn, y_tst

    def expandDims(self, x):
        x = np.expand_dims(x, axis=-1)
        return x

    def normalizeNormX(self, x):
        ss = pp.MinMaxScaler((-1,1))

        x = ss.fit_transform(x)
        
        return x

    def normalizeNormY(self, y):
        y = (y - y.min()) / ( y.max() - y.min() )
        return y
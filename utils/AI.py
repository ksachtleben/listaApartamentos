from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization, Conv1D, MaxPool1D, Dropout, Input, LSTM, AveragePooling1D, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal


from pickle import dump



class AI:
    def __init__(self):
        self._verbose = 1
        self._epochs = 50000
        self._batchSize = 32
        self._bits = 128

    def createModel(self, xTrain, xTest, yTrain, yTest, mean, var):

        nSteps, nFeatures, nOutputs = xTrain.shape[1], 1, 1
        
        verbose = self._verbose
        epochs = self._epochs
        batch_size = self._batchSize
        bits = self._bits

        model = Sequential()

        
        '''model.add(Conv1D(filters=bits, kernel_size=2, activation='elu', padding='same'))
        model.add(MaxPool1D(pool_size=2,strides=2,padding='valid'))
        
        model.add(Conv1D(filters=2*bits, kernel_size=2, activation='elu', padding='same'))
        model.add(MaxPool1D(pool_size=2,strides=2,padding='valid'))
        
        model.add(LSTM(4*bits, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(8*bits, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))'''
        #model.add(LSTM(4*bits, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(nSteps,nFeatures)))
        initializer = RandomNormal(mean=mean, stddev=var)       

        model.add(Dense(bits, activation='relu', input_shape=(nSteps,)))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(bits, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(nOutputs, activation='softplus', kernel_initializer=initializer))

        '''model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=1.0,
                                                                beta_1=0.99,
                                                                epsilon=1e-6,
                                                                decay=1e-2), metrics=['accuracy'])'''
        model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])

        print(model.summary())

        es = EarlyStopping(monitor='loss',patience=50, mode='min', restore_best_weights=True, verbose=0)

        model.fit(xTrain,yTrain,validation_data=(xTest,yTest), batch_size=batch_size, epochs=epochs, callbacks=[es], verbose=verbose)

        model.save('/home/kewin/apartamentos/models')


        _, accuracy = model.evaluate(xTest,yTest, batch_size=batch_size, verbose=verbose)

        return model, accuracy
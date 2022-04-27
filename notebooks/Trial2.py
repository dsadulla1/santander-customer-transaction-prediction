from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    '''
    Data providing function:

    Make sure to have every relevant import statement included here and return data as
    used in model function below. This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import numpy
    
    df = pd.read_csv("../data/train.csv", sep=",")
    X = df[df.columns[2:]].astype(float).values
    Y = np.array(df['target']).astype(float)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    X_train, Y_train, X_test, Y_test = X[:-60000,:], Y[:-60000], X[-60000:,:], Y[-60000:]
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
    from keras.regularizers import l1_l2

    model = Sequential()
    model.add(Dense({{choice([256, 512, 1024])}}, 
                    input_shape=(X_train.shape[1],),
                    kernel_regularizer = l1_l2(l1={{uniform(0, 1)}}, 
                                               l2={{uniform(0, 1)}})))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense({{choice([256, 512, 1024])}},
                    kernel_regularizer = l1_l2(l1={{uniform(0, 1)}}, 
                                               l2={{uniform(0, 1)}})))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense({{choice([256, 512, 1024])}},
                    kernel_regularizer = l1_l2(l1={{uniform(0, 1)}}, 
                                               l2={{uniform(0, 1)}})))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
    
    #MCP = ModelCheckpoint(filepath = "../results/BestModel1.h5", monitor='val_loss', verbose=0, save_best_only=True, 
    #                  save_weights_only=False, mode='auto', period=1)
    
    #RLROP = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, 
    #                      mode='auto', min_delta=0.00001, cooldown=1, min_lr=0)
    
    #CSVL = CSVLogger(filename = "../results/LogFile1.txt", separator=',', append=False)
    
    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              epochs=5,
              
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    import h5py
    from sklearn.metrics import roc_auc_score
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    best_model.save('../results/Trial2.h5')

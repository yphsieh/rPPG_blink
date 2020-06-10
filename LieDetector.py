import os
import time
import argparse
import numpy as np
import pandas as pd

from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.metrics import Recall
from sklearn.model_selection import train_test_split

from data_preprocessing import *

parser = argparse.ArgumentParser(description='')
# actions
parser.add_argument('--train',  action='store_true', default=False)
parser.add_argument('--test',   action='store_true', default=False)
# parameters for data preprocessing
parser.add_argument('--seed',   type=int,   default=22)
parser.add_argument('--smooth', action='store_true',  default=False)
parser.add_argument('--scale',  action='store_true',  default=False)
# directory
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./save')
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

data = np.load(os.path.join(args.data_dir, 'data_600.npy'))
label = np.load(os.path.join(args.data_dir, 'label_600.npy')).reshape(-1, 1)
# general, video = data[0], data[1]

x_train, x_val, y_train, y_val = DataPreprocess(data, label, 0.2, args.seed, smooth=args.smooth, scale=args.scale)

# print('x_train', x_train[:3])
# print('x_val', x_val[:3])

if args.train:
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(600,2)))
    model.add(LSTM(units=32, return_sequences=True, input_shape=(600, 2)))
    # model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2)))
    # model.add(Dense(units=128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()
    model_acc = model

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Recall()])

    callbacks = []
    callbacks.append(ModelCheckpoint(os.path.join(args.save_dir,'RDNN_s'+str(args.seed)+'.h5'), monitor='recall_1', verbose=1, save_best_only=True, mode='max'))
    csv_logger = CSVLogger(os.path.join(args.save_dir, 'RDNN_s'+str(args.seed)+'_log.csv'), separator=',', append=False)
    callbacks.append(csv_logger)
    earlystop = EarlyStopping(monitor='recall_1', patience=5, mode='max')
    callbacks.append(earlystop)

    tStart = time.time()
    model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), shuffle=True, batch_size=256, callbacks=callbacks)
    print('Costing time: ', time.time()-tStart, ' ......')

    ## accuracy
    model = model_acc
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []
    callbacks.append(ModelCheckpoint(os.path.join(args.save_dir,'RDNN_s'+str(args.seed)+'.h5'), monitor='accuracy', verbose=1, save_best_only=True, mode='max'))
    csv_logger = CSVLogger(os.path.join(args.save_dir, 'RDNN_s'+str(args.seed)+'_log_acc.csv'), separator=',', append=False)
    callbacks.append(csv_logger)
    earlystop = EarlyStopping(monitor='accuracy', patience=5, mode='max')
    callbacks.append(earlystop)

    tStart = time.time()
    model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), shuffle=True, batch_size=256, callbacks=callbacks)
    print('Costing time: ', time.time()-tStart, ' ......')


if args.test:
    os.system('python evaluation.py -m '+ os.path.join(args.save_dir,'RDNN_s'+str(args.seed)+'.h5') + (' --smooth ' if args.smooth) + (' --scale ' if args.scale) )


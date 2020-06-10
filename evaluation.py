import os
import argparse
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', default='save/RDNN.h5', type=str)
parser.add_argument('--smooth', type=bool, default=False)
parser.add_argument('--scale', type=bool, default=False)
args = parser.parse_args()
print(args)


x_test = np.load('data/data_test_600.npy')
y_test = np.load('data/label_test_600.npy').reshape(-1, 1)
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

lie_ratio = np.sum(y_test)/y_test.shape[0]
print('Lie Ratio: {}'.format(lie_ratio))

x_test = TestPreprocess(x_test, args.smooth, args.scale)

print('='*20, 'Model Loading...', '='*20)
model = load_model(args.model_name)
print('='*20, 'Model Loaded', '='*20)

# os.system('clear')

predict = model.predict(x_test)
y_predict = (predict > 0.3).astype(np.int)

lie_ratio = np.sum(y_predict)/y_predict.shape[0]
print('Lie Ratio Predicted: {}'.format(lie_ratio))


score_f1 = f1_score(y_test, y_predict)
score_acc = accuracy_score(y_test, y_predict)
print('f1 score: {}'.format(score_f1))
print('accuracy score: {}'.format(score_acc))

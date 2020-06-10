import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def DataPreprocess(data, label, ts, rdm, smooth, scale):
    x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=ts, random_state=rdm)
    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_val shape: {}'.format(x_val.shape))
    print('y_val shape: {}'.format(y_val.shape))

    lie_ratio_train = 1 - np.sum(y_train)/y_train.shape[0]
    print('Lie Ratio Train: {}'.format(lie_ratio_train))
    lie_ratio_val = 1 - np.sum(y_val)/y_val.shape[0]
    print('Lie Ratio Val: {}'.format(lie_ratio_val))

    x_axis = np.linspace(0, len(x_train[0]), len(x_train[0])) 
    if not scale: plt.plot(x_axis, x_train[rdm*10]) 

    if smooth:
        for i in range(len(x_train)):
            tmp = [x_train[i][j][0] for j in range(len(x_train[0])) if x_train[i][j][0] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)): x_train[i][j][0] = tmp.values[j]

            tmp = [x_train[i][j][1] for j in range(len(x_train[0])) if x_train[i][j][1] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)): x_train[i][j][1] = tmp.values[j]

        for i in range(len(x_val)):
            tmp = [x_val[i][j][0] for j in range(len(x_val[0])) if x_val[i][j][0] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)): x_val[i][j][0] = tmp.values[j]

            tmp = [x_val[i][j][1] for j in range(len(x_val[0])) if x_val[i][j][1] != 0]
            tmp = pd.DataFrame(tmp)
            tmp = tmp.ewm(span=300).mean()
            for j in range(len(tmp.values)): x_val[i][j][1] = tmp.values[j]


    if scale:
        for i in range(len(x_train)):
            ave = [x_train[i][j][0] for j in range(len(x_train[0])) if x_train[i][j][0] != 0]
            ave = np.average(ave)
            x_train[i] /= ave

        for i in range(len(x_val)):
            ave = [x_val[i][j][0] for j in range(len(x_val[0])) if x_val[i][j][0] != 0]
            x_val[i] /= np.average(ave)

    

    plt.plot(x_axis, x_train[rdm*10])
    if not scale: plt.legend(labels=['original general','original data','processed general', 'processed data'])
    else : plt.legend(labels=['processed general', 'processed data'])

    plt.savefig('temp.png')

   
    return x_train, x_val, y_train, y_val

def TestPreprocess(x_test, smooth, scale):
    if smooth:
        for i in range(len(x_test)):
            tmp = pd.DataFrame(x_test[i])
            tmp.ewm(span=300).mean()
            x_test[i] = tmp.values

    if scale:
        for i in range(len(x_test)):
            ave = [x_test[i][j][0] for j in range(len(x_test[0])) if x_test[i][j][0] != 0]
            ave = np.average(ave)
            x_test[i] /= ave

    return x_test

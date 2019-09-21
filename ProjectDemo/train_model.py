from baseline_classifiers import svm_classifier, dt_classifier, gb_classifier, knn_classifier
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

# loading data
print ("Reading the raw data...")
df = pd.read_csv('C:\\Users\\JYAO09\\Documents\\EE5111\\ProjectDemo\\Data\\ecg_mh.csv', header=None)
print(df.shape)

# normalization
print ('Start normalization...')
for row in range(0, 160):
    for col in range(0, 15):
        df.loc[row, (col*600):((col+1)*600)] = np.reshape(scaler.fit_transform(np.reshape(df.loc[row, (col*600):((col+1)*600)].values, (-1,1))), (-1,))

print ('Down normalization!')

#EEG Label
print ("Loading labels...")
# ECG label
label = pd.read_csv('C:\\Users\\JYAO09\\Documents\\EE5111\\ProjectDemo\\Data\\label_mh.csv', header=None)
label = label.iloc[:,-1]


print ("Classification starts...")
svm_classifier(df, label, runs=50)
dt_classifier(df, label, runs=50)
gb_classifier(df, label, runs=50)
knn_classifier(df, label, runs=50)

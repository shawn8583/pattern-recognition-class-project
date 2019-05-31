import csv
import glob
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

training_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv.csv'

df = pd.read_csv(training_data_csv, sep=',', header=None)
# print(df.values)

train_data = []
train_data = np.matrix(df.values)
shape = train_data.shape
print('***************************************************************')
print('Training data is a %s matrix:' %(train_data.shape,))
print(train_data)
print('***************************************************************')
print('\n')

# train_data = np.array(df.values)

# Normalize matrix and apply PCA, result in reducing the vector space to 40 out of 99 dimensions
model = PCA(n_components=40)
pts = normalize(train_data)
model.fit(pts)
pts2 = model.transform(pts)   # transform() is to standerdize through finding the center

print('***************************************************************')
print('AFTER NORMALIZATION AND PCA, Training data is a %s matrix:' %(pts2.shape,))
print(pts2)
print('***************************************************************')
print('\n')

# display eigenfaces
# fig, axes = plt.subplots(1, 40)
# for i in range(40):
#       ord = train_data.components_[i]
#       img = ord.reshape(10*10)
#       ax[i].imshow(img, cmap='gray')


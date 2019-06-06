#!/usr/bin/python

import os
import csv
import math
import time
import numpy as np 
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# reading eigenface data
training_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR'  
training_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'      # This is a path in my Ubuntu Linux
testing_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS'
label_training = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR'
label_training_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR_csv'

train_rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'
train_newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv.csv'
label_rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR_csv'
label_newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR_csv.csv'

# # create "faceR_csv.csv" file for csv format data
# os.mknod(training_data_csv)
# os.mknod(label_csv)

# define header for csv, named 0 - 99
header_for_training_csv = []
header_for_training_label = ['#', 'sex', 'age', 'race', 'face', 'prop']
header_temp = []
for i in range(0, 100):
      header_temp.append(i)
header_for_training_csv = [str(i) for i in header_temp]    # convert numbers to string type as there are all strings in csv file
# print(header)

# converting faceR (training data) to csv format
if os.path.exists(train_newname) == False:
      os.mknod(training_data_csv)   # create "faceR_csv.csv" file for csv format data
      with open(training_data, 'r+') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                  lines[i] = lines[i].lstrip()  # There are spaces as the first character in the oroginal data file, .lstrip will delete the spaces
                  lines[i] = lines[i].replace('  ', ',')     # replacing "space" with "comma" makes the data in csv format
                  lines[i] = lines[i].replace(' ', ',')
      with open(training_data_csv, 'r+') as training_csv:   # write changes
            writer = csv.writer(training_csv)
            writer.writerow(header_for_training_csv)       # write header line in the first row
            training_csv.writelines(lines)      # write other data lines
      os.rename(train_rename, train_newname)    # add filename extension '.csv'
      # Here we have a complete csv format eigenface data file named 'faceR_csv.csv'
else:
      print("########### The csv format training data already exists! ############")
      print('\n')

# Converting faceDR (label for training data) to csv format
if os.path.exists(label_newname) == False:
      os.mknod(label_training_csv)
      with open(label_training, 'r+') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                  lines[i] = lines[i].lstrip()
                  lines[i] = lines[i].replace('_sex  ', '')
                  lines[i] = lines[i].replace('_age  ', '')
                  lines[i] = lines[i].replace('_race ', '')
                  lines[i] = lines[i].replace('_face ', '')
                  lines[i] = lines[i].replace("(_prop '", "")      # double quotes and single quotes in python can both be used to represent string.
                  lines[i] = lines[i].replace('hat ', 'hat_')
                  lines[i] = lines[i].replace('moustache ', 'moustache_')
                  lines[i] = lines[i].replace('glasses ', 'glasses_')
                  lines[i] = lines[i].replace('bandana ', 'bandana_')
                  lines[i] = lines[i].replace('beard ', 'beard_')
                  lines[i] = lines[i].replace('(_missing descriptor)', '(missing_descriptor)')
                  lines[i] = lines[i].replace('_)', '')
                  lines[i] = lines[i].replace('))', ')')
                  lines[i] = lines[i].replace('()', '(\)')
                  lines[i] = lines[i].replace('  ', ',')
                  lines[i] = lines[i].replace(' ', ',')
      with open(label_training_csv,'r+') as label_training_csv:
            writer = csv.writer(label_training_csv)
            writer.writerow(header_for_training_label)
            label_training_csv.writelines(lines)
      os.rename(label_rename, label_newname)
else:
      print("########### The csv format training label already exists! ###########")
      print('\n')


# ----------------------- Append labels to training data --------------------
train = pd.read_csv(train_newname, sep=',', header=0) # 'jeader' sets which row as index for columns/ 'index_col' sets which column as index for rows
label = pd.read_csv(label_newname, sep=',', header=0)
race = label.iloc[:,3]
# training_data_with_label = train.merge(race)  # cannot use merge or join here
training_data_with_labels = pd.concat([train, race], axis=1)
print('INFO: Labels of "RACE" have been appended to the last column of training data')
print('INFO: Now Training data is a Dataframe with %s rows x %s columns' %(training_data_with_labels.shape[0], training_data_with_labels.shape[1]))
# print(training_data_with_labels)


# -------------------------- Deleting Missing Rows ----------------------------
df = training_data_with_labels
missing_rows = []
cleared_training_data = df
for i in range(0, 1999):
      sum = 0
      check_empty = df.iloc[i,:]
      if isinstance(check_empty[100], float):
            missing_rows.append(i)
      else:
            for a in range(1, 99):
                  sum = sum + check_empty[a]
                  if sum == 0 :
                        if a == 2:  # 2 is random selected number, if will append 'a' 99 times otherwise
                              missing_rows.append(i)

print('INFO: These rows have NaN labels ot zero data: %s' %(missing_rows))

# drop rows with missing data
a = 0
for i in missing_rows:
      cleared_training_data = cleared_training_data.drop(cleared_training_data.index[i-a])
      a = a + 1   # having a to solve index problems after deleting rows
      print('INFO: Deleted Row %d' %(i,))
cleared_training_data = cleared_training_data.reset_index(drop=True)    # reset row index that has become inconsecutive after deleting missing rows
print('INFO: Now After clearing missing data, Training data is a Dataframe with [%s rows x %s columns]' %(cleared_training_data.shape[0], cleared_training_data.shape[1]))
print('\n')
print('----------------------------------------- Cleared Training Data -------------------------------------------')
print(cleared_training_data)
print('-----------------------------------------------------------------------------------------------------------')
print('\n')
print('--------- Training Dataset info ---------')
print(cleared_training_data.info())
print('-----------------------------------------')
print('Number of samples in each label:')
print(cleared_training_data['race'].value_counts())
print('-----------------------------------------')

# ------------------------ Plots ------------------------
df = cleared_training_data
# plot features in a pair plot, will plot 99*99 figures
# tmp = df.drop('0', axis=1)
# g = sns.pairplot(tmp, hue='race', markers='+')
# plt.show()

g = sns.violinplot(y='race', x='50', data=df, inner='quartile')   # plot a violin plot with feature 27

X = df.drop(['0', 'race'], axis=1)
y = df['race']

# Plot 2D using TSNE
tsne = TSNE(n_components=3, random_state=0, n_iter=5000)
X_3d = tsne.fit_transform(X)
label_ids = ['(white)', '(black)', '(asian)', '(hispanic)', '(other)']
colors = 'r', 'g', 'b', 'c', 'm'
mid = int(len(X_3d)/2)
fig = plt.figure()
print(mid)
ax = fig.add_subplot(111, projection='3d')
for i, c, label in zip(label_ids, colors, label_ids):
      ax.scatter(X_3d[y == i, 0][0: ], X_3d[y==i, 1][0: ], X_3d[y==i,2][0: ], c=c, s=100, label=label, marker='+')
      # ax.scatter(X_3d[y == i, 0][mid:], X_3d[y==i, 1][mid:], X_3d[y==i,2][mid:], c=c, s=100, label=label, marker='.')
# plt.scatter(X_2d[y=='(white)', 0], X_2d[y=='(white)', 1], c=y, label='white')
# plt.scatter(X_2d[y=='(black)', 0], X_2d[y=='(black)', 1], c=b, label='black')
# plt.scatter(X_2d[y=='asian', 0], X_2d[y=='asian', 1], c=r, label='asian')
# plt.scatter(X_2d[y=='hispanic', 0], X_2d[y=='hispanic', 1], c=m, label='hispanic')
# plt.scatter(X_2d[y=='other', 0], X_2d[y=='other', 1], c=g, label='other')
plt.legend()
plt.show()




# ------------------ Applying Multi-layer Perceptron ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))

def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == y_test)/len(y_test)))
    time.sleep(0.5) # to let the print get out before any progress bars

nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
nn.fit(X_train, y_train)
print_accuracy(nn.predict)



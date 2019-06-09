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
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# reading eigenface data
training_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR'  
training_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'      # This is a path in my Ubuntu Linux
testing_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS'
testing_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS_csv'
label_training = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR'
label_training_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR_csv'
label_testing = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDS'
label_testing_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDS_csv'

train_rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'
train_newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv.csv'
label_train_rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR_csv'
label_train_newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDR_csv.csv'
test_rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS_csv'
test_newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS_csv.csv'
label_test_rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDS_csv'
label_test_newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceDS_csv.csv'

# ---------------- Converting Original Data to CSV format -----------------
def convert_data_to_csv(original_file_path, csv_file_path, old_file_name, new_file_name):
      header_for_data = []    # define header for csv, named 0 - 99
      header_temp = []
      for i in range(0, 100):
            header_temp.append(i)
      header_for_data = [str(i) for i in header_temp]    # convert numbers to string type as there are all strings in csv file
      # print(header)

      # converting faceR (training data) to csv format
      if os.path.exists(new_file_name) == False:
            os.mknod(csv_file_path)   # create "faceR_csv.csv" file for csv format data
            with open(original_file_path, 'r+') as f:
                  lines = f.readlines()
                  for i in range(0, len(lines)):
                        lines[i] = lines[i].lstrip()  # There are spaces as the first character in the oroginal data file, .lstrip will delete the spaces
                        lines[i] = lines[i].replace('  ', ',')     # replacing "space" with "comma" makes the data in csv format
                        lines[i] = lines[i].replace(' ', ',')
            with open(csv_file_path, 'r+') as training_csv:   # write changes
                  writer = csv.writer(training_csv)
                  writer.writerow(header_for_data)       # write header line in the first row
                  training_csv.writelines(lines)      # write other data lines
            os.rename(old_file_name, new_file_name)    # add filename extension '.csv'
            # Here we have a complete csv format eigenface data file named 'faceR_csv.csv'
      else:
            print("########### The csv format training data already exists! ############")
            print('\n')


def convert_label_to_csv(original_label_file_path, csv_label_file_path, old_label_file_name, new_label_file_name):
      # Converting faceDR (label for training data) to csv format
      header_for_label = ['#', 'sex', 'age', 'race', 'face', 'prop']
      if os.path.exists(new_label_file_name) == False:
            os.mknod(csv_label_file_path)
            with open(original_label_file_path, 'r+') as f:
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
            with open(csv_label_file_path,'r+') as csv_label_file_path:
                  writer = csv.writer(csv_label_file_path)
                  writer.writerow(header_for_label)
                  csv_label_file_path.writelines(lines)
            os.rename(old_label_file_name, new_label_file_name)
      else:
            print("########### The csv format training label already exists! ###########")
            print('\n')

convert_data_to_csv(training_data, training_data_csv, train_rename, train_newname)  # training data to csv
convert_data_to_csv(testing_data, testing_data_csv, test_rename, test_newname)  #testing data to csv

convert_label_to_csv(label_training, label_training_csv, label_train_rename, label_train_newname)  # training label to csv
convert_label_to_csv(label_testing, label_testing_csv, label_test_rename, label_test_newname)  # testing label to csv


# ----------------------- Append labels to training data --------------------
train = pd.read_csv(train_newname, sep=',', header=0) # 'header' sets which row as index for columns/ 'index_col' sets which column as index for rows
test = pd.read_csv(test_newname, sep=',', header=0)
train_label = pd.read_csv(label_train_newname, sep=',', header=0)
test_label = pd.read_csv(label_test_newname, sep=',', header=0)
train_race = train_label.iloc[:,3]
test_race = test_label.iloc[:,3]

training_data_with_labels = pd.concat([train, train_race], axis=1)
testing_data_with_labels = pd.concat([test, test_race], axis=1)
# training_data_with_label = train.merge(race)  # cannot use merge or join here
print('INFO: Labels of "RACE" have been appended to the last column of training amd testing data')
print('INFO: Now Training data is a Dataframe with %s rows x %s columns' %(training_data_with_labels.shape[0], training_data_with_labels.shape[1]))
print('INFO: Now Testing data is a Dataframe with %s rows x %s columns' %(testing_data_with_labels.shape[0], testing_data_with_labels.shape[1]))
# print(training_data_with_labels)

# --------------- Deleting Missing Rows -------------
def deleting_missing_rows(df, cleared_data):
      missing_rows = []
      cleared_data = df
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
            cleared_data = cleared_data.drop(cleared_data.index[i-a])
            a = a + 1   # having a to solve index problems after deleting rows
            print('INFO: Deleted Row %d' %(i,))
      cleared_data = cleared_data.reset_index(drop=True)    # reset row index that has become inconsecutive after deleting missing rows
      return cleared_data

cleared_training_data = training_data_with_labels
cleared_testing_data = testing_data_with_labels

cleared_training_data = deleting_missing_rows(training_data_with_labels, cleared_training_data)
cleared_testing_data = deleting_missing_rows(testing_data_with_labels, cleared_testing_data)

print('INFO: Now After clearing missing data, Training data is a Dataframe with [%s rows x %s columns]' %(cleared_training_data.shape[0], cleared_training_data.shape[1]))
print('INFO: Now After clearing missing data, Testing data is a Dataframe with [%s rows x %s columns]' %(cleared_testing_data.shape[0], cleared_testing_data.shape[1]))
print('\n')
print('----------------------------------------- Cleared Training Data -------------------------------------------')
print(cleared_training_data)
print('-----------------------------------------------------------------------------------------------------------')
print('\n')
print('----------------------------------------- Cleared Testing Data -------------------------------------------')
print(cleared_testing_data)
print('-----------------------------------------------------------------------------------------------------------')
print('\n')
print('--------- Training Dataset info ---------')
print(cleared_training_data.info())
print('-----------------------------------------')
print('Number of samples in each label:')
print(cleared_training_data['race'].value_counts())
print('-----------------------------------------')
print('\n')
print('--------- Training Dataset info ---------')
print(cleared_testing_data.info())
print('-----------------------------------------')
print('Number of samples in each label:')
print(cleared_testing_data['race'].value_counts())
print('-----------------------------------------')

# ------------------------ Plots ------------------------
df_train = cleared_training_data
df_test = cleared_testing_data
# plot features in a pair plot, will plot 99*99 figures
# tmp = df.drop('0', axis=1)
# g = sns.pairplot(tmp, hue='race', markers='+')
# plt.show()

g = sns.violinplot(y='race', x='50', data=df_train, inner='quartile')   # plot a violin plot with feature 27

X_train = df_train.drop(['0', 'race'], axis=1)
y_train = df_train['race']
X_test = df_test.drop(['0', 'race'], axis=1)
y_test = df_test['race']

# ------------------ Plot 2D using TSNE ---------------
# tsne = TSNE(n_components=2, random_state=0)
# X_2d = tsne.fit_transform(X)
# label_ids = ['(white)', '(black)', '(asian)', '(hispanic)', '(other)']
# plt.figure(figsize=(10, 5))
# colors = 'r', 'g', 'b', 'c', 'm'
# for i, c, label in zip(label_ids, colors, label_ids):
#       plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label, marker='+')

# ----------------- Plot 3D using TSNE --------------
# tsne = TSNE(n_components=3, random_state=0, n_iter=5000)
# X_3d = tsne.fit_transform(X)
# label_ids = ['(white)', '(black)', '(asian)', '(hispanic)', '(other)']
# colors = 'r', 'g', 'b', 'c', 'm'
# mid = int(len(X_3d)/2)
# fig = plt.figure()
# print(mid)
# ax = fig.add_subplot(111, projection='3d')
# for i, c, label in zip(label_ids, colors, label_ids):
#       ax.scatter(X_3d[y == i, 0][0: ], X_3d[y==i, 1][0: ], X_3d[y==i,2][0: ], c=c, s=100, label=label, marker='+')
plt.legend()
plt.show()

# ------------------ Normalization and Applying PCA -------------------
def normalization_and_PCA(dimention, df):
      df_matrix = np.matrix(df.values)
      model = PCA(n_components=dimention)
      pts = normalize(df_matrix)
      model.fit(pts)
      pts2 = model.transform(pts)
      return pts

# applying to X_train, X_test, y_train, y_test
# X_train = normalization_and_PCA(40, X_train)
# X_test = normalization_and_PCA(40, X_test)
# y_train = normalization_and_PCA(40, y_train)
# y_test = normalization_and_PCA(40, y_test)

# ------------------ Applying Multi-layer Perceptron ------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print('There are {} samples in the training set and {} samples in the test set'.format(
# X_train.shape[0], X_test.shape[0]))

def print_accuracy(f):
      print('\n')
      print('------------- Multi-layer Perceptron ------------')
      print("Right Classification Samples: {0}".format(np.sum(f(X_test)==y_test)))
      print("Wrong Classification Samples: {0}".format(np.sum(f(X_test)!=y_test)))
      print("Accuracy for Multi-layer Perceptron = {0}%".format(100*np.sum(f(X_test) == y_test)/len(y_test)))
      time.sleep(0.5) # to let the print get out before any progress bars
      print('-------------------------------------------------')
      print('\n')

nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(500,200), random_state=0)
nn.fit(X_train, y_train)
print_accuracy(nn.predict)


# ------------------ Applying K-Nearest Neighbors ------------------
k_range = list(range(1,26))
scores = []

k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

knn_accuracy = 100*metrics.accuracy_score(y_test, y_pred)
knn_accuracy = round(knn_accuracy, 1)
print("Accuracy of KNN is {0}%".format(knn_accuracy,))
#!/usr/bin/python

import os
import csv
import numpy as np 
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd


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
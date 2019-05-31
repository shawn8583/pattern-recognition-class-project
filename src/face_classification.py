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
label_

rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'
newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv.csv'

# create "faceR_csv.csv" file for csv format data
os.mknod(training_data_csv)
os.mknod(label_csv)

# define header for csv, named 0 - 99
header = []
header_temp = []
for i in range(0, 100):
      header_temp.append(i)
header = [str(i) for i in header_temp]    # convert numbers to string type as there are all strings in csv file
# print(header)

# converting faceR (training data) to csv format
with open(training_data, 'r+') as f:
      lines = f.readlines()
      for i in range(0, len(lines)):
            lines[i] = lines[i].lstrip()  # There are spaces as the first character in the oroginal data file, .lstrip will delete the spaces
            lines[i] = lines[i].replace('  ', ',')     # replacing "space" with "comma" makes the data in csv format
            lines[i] = lines[i].replace(' ', ',')

with open(training_data_csv, 'r+') as training_csv:   # write changes
      writer = csv.writer(training_csv)
      writer.writerow(header)       # write header line in the first row
      training_csv.writelines(lines)      # write other data lines

os.rename(rename, newname)    # add filename extension '.csv'
# Here we have a complete csv format eigenface data file named 'faceR_csv.csv'

# Converting faceDR (label for training data) to csv format
with open()
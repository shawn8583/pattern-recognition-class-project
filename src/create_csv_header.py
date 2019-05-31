#!/usr/bin/python

import os
import csv
import numpy as np 
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd


# reading eigenface data
training_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR'  
training_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv_test'      # This is a path in my Ubuntu Linux
testing_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS'

rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv_test'
newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv_test.csv'

# create "faceR_csv_test_test.csv" file for csv format data
os.mknod(training_data_csv)

header = []
header_temp = []
for i in range(0, 100):
      header_temp.append(i)
header = [str(i) for i in header_temp]
print(header)

# converting the data to csv format
with open(training_data, 'r+') as f:
      lines = f.readlines()
      for i in range(0, len(lines)):
            lines[i] = lines[i].lstrip()  # There are spaces as the first character in the oroginal data file, .lstrip will delete the spaces
            lines[i] = lines[i].replace('  ', ',')     # replacing "space" with "comma" makes the data in csv format
            lines[i] = lines[i].replace(' ', ',')

with open(training_data_csv, 'r+') as training_csv:   # write changes
      writer = csv.writer(training_csv)
      writer.writerow(header)
      training_csv.writelines(lines)

os.rename(rename, newname)    # add filename extension '.csv'
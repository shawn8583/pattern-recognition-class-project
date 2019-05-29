#!/usr/bin/python

import os
import numpy as np 
import matplotlib as mpl
from matplotlib import pyplot as plt


# reading eigenface data
training_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR'  
training_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'      # This is a path in my Ubuntu Linux
testing_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS'

rename = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'
newname = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv.csv'

# create "faceR_csv.csv" file for csv format data
os.mknod(training_data_csv)

# converting the data to csv format
with open(training_data, 'r+') as f:
      lines = f.readlines()
      for i in range(0, len(lines)):
            lines[i] = lines[i].lstrip()  # There are spaces as the first character in the oroginal data file, .lstrip will delete the spaces
            lines[i] = lines[i].replace('  ', ',')     # replacing "space" with "comma" makes the data in csv format
            lines[i] = lines[i].replace(' ', ',')

with open(training_data_csv, 'r+') as training_csv:   # write changes
      training_csv.writelines(lines)

os.rename(rename, newname)    # add filename extension '.csv'



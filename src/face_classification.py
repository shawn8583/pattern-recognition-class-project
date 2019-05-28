#!/usr/bin/python

import numpy as np 
import matplotlib as mpl
from matplotlib import pyplot as plt


# reading eigenface data
training_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR'  
training_data_csv = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv'      # This is a path in my Ubuntu Linux

testing_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceS'

# converting the data to csv format
# faceR = open(training_data, "a+")
with open(training_data, 'r+') as f:
      lines = f.readlines()
      for i in range(1, len(lines)):
            lines[i] = lines[i].strip()

with open(training_data_csv, 'r+') as training_csv:
      training_csv.writelines(lines)
            
# csv_faceR = txt_faceR.replace(' ', ',') # replace space by comma
# faceR.close()

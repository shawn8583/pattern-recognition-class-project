#!/usr/bin/python

import os
import csv
import math
import time
import numpy as np 
import pandas as pd

def convert_data_to_csv(original_file_path, csv_file_path, old_file_name, new_file_name):
      header_for_data = []
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
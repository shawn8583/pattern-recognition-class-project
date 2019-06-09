import csv
import glob
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def normalization_and_PCA(csv_file_path, dimention, df, df_after_pca):
      df = pd.read_csv(csv_file_path, sep=',', header=0)
      df_matrix = np.matrix(df.values)
      model = PCA(n_components=dimention)
      pts = normalize(df_matrix)
      model.fit(pts)
      pts2 = model.transform(pts)
      return df_after_pca
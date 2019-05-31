import pandas as pd

training_data = '/home/shawn/projects/pattern_recognition_class_project/face/csv_eigenfaces/faceR_csv.csv'

df = pd.read_csv(training_data)
print(df.head(3))

### K-means clustering


# Import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn import decomposition
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Read data
url='https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'
df_raw = pd.read_csv(url)


# Select columns
df = df_raw.copy()
df = df[['Latitude', 'Longitude', 'MedInc']]


# Scale data
scaler = StandardScaler()
df_scale = scaler.fit_transform(df)


# K-means
kmeans = KMeans(init="random",n_clusters=2, random_state=0, n_init=10,max_iter=300)
kmeans.fit(df_scale)

# Return to the original variables
df_2 = scaler.inverse_transform(df_scale)


# Add column with number of cluster
df_2=pd.DataFrame(df_2,columns=['Latitude','Longitude','MedInc'])
df_2['Cluster'] = kmeans.labels_

# Convert cluster to categorical 
df_2['Cluster'] = pd.Categorical(df_2.Cluster)

# load the model from disk
filename = '../models/finalized_model.sav' #use absolute path
loaded_model = pickle.load(open(filename, 'rb'))

# get the predict : means which cluster it belongs to
print('Predicted ] : \n', loaded_model.predict([[ 2.344766,  1.052548,  -1.327835], [ -0.044727,0.542225, 0.329279]])) 
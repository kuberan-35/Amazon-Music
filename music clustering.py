import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv("single_genre_artists.csv")   
print("Dataset Shape:", df.shape)
print(df.head())

df_features = df.drop(columns=['track_id', 'track_name', 'artist_name'], errors='ignore')

df_features = df_features.select_dtypes(include=['float64', 'int64'])

if set(['track_id','track_name','artist_name']).issubset(df.columns):
    df = df.drop(['track_id','track_name','artist_name'], axis=1)
    
df = df.dropna()

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

te_array = te.fit(df).transform(df).astype('int')

te_array

df_features = df_features.fillna(df_features.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

inertia = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(pca_features[:,0], pca_features[:,1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Clusters Visualization (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()

cluster_profile = df.groupby('Cluster').mean()
print(cluster_profile)

plt.figure(figsize=(10,6))
sns.heatmap(cluster_profile.T, cmap='coolwarm', annot=True)
plt.title('Cluster Feature Profiles')
plt.show()

df.to_csv("Amazon_Music_Clusters.csv", index=False)
print("Clustered dataset saved as Amazon_Music_Clusters.csv")
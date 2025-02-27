import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import streamlit as st

# Încărcarea setului de date
file_path = r"C:\Users\Vladislav\Desktop\laboratoare\files\bank+marketing\bank\bank-full.csv"
df = pd.read_csv(file_path, sep=';')

# Interfață Streamlit
st.title("Clustering și Reducerea Dimensionalității - Bank Marketing")

# Previzualizare date
st.write("Primele 5 rânduri din setul de date:")
st.dataframe(df.head())

# Statistici descriptive
st.write("Statistici descriptive ale setului de date:")
st.dataframe(df.describe())

# Tratarea valorilor lipsă
st.write("Valori lipsă per coloană:")
st.dataframe(df.isnull().sum().to_frame(name='Missing Values'))
df.dropna(inplace=True)

# Conversia variabilelor categorice în numerice (one-hot encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# Scalarea datelor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Reducerea dimensionalității cu PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Aplicarea clustering-ului K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Aplicarea clustering-ului DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

# Evaluarea modelelor cu scorul Silhouette
silhouette_kmeans = silhouette_score(X_pca, kmeans_labels)
silhouette_dbscan = silhouette_score(X_pca, dbscan_labels)

# Afișarea rezultatelor modelelor
st.write("Compararea performanțelor modelelor de clustering:")
results = pd.DataFrame({
    "Model": ["K-Means", "DBSCAN"],
    "Silhouette Score": [silhouette_kmeans, silhouette_dbscan]
})
st.dataframe(results)

# Vizualizarea clusterelor
st.write("Vizualizarea clusterelor K-Means")
fig_kmeans, ax_kmeans = plt.subplots()
ax_kmeans.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis', alpha=0.5)
ax_kmeans.set_title('K-Means Clustering')
st.pyplot(fig_kmeans)

st.write("Vizualizarea clusterelor DBSCAN")
fig_dbscan, ax_dbscan = plt.subplots()
ax_dbscan.scatter(X_pca[:,0], X_pca[:,1], c=dbscan_labels, cmap='viridis', alpha=0.5)
ax_dbscan.set_title('DBSCAN Clustering')
st.pyplot(fig_dbscan)

if __name__ == "__main__":
    st.write("Aplicația Streamlit este pornită. Accesează localhost pentru vizualizare.")
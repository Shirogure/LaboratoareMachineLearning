import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Încărcare set de date
fisier_date = r"C:\Users\Vladislav\Desktop\laboratoare\files\bike+sharing+dataset\day.csv"
date_biciclete = pd.read_csv(fisier_date)

# Titlul aplicației
st.title("Explorare și modelare: Bike Sharing")
st.write("Vizualizare date brute:")
st.dataframe(date_biciclete.head())

# Statistici sumare
st.subheader("Analiza descriptivă")
st.dataframe(date_biciclete.describe())

# Identificarea valorilor lipsă
valori_lipsa = date_biciclete.isnull().sum().to_frame(name="Număr valori lipsă")
st.write("Tabel cu valori lipsă:")
st.dataframe(valori_lipsa)

# Eliminare valori lipsă (dacă există)
date_biciclete.dropna(inplace=True)

# Detectare outlieri prin boxplot
st.subheader("Distribuția variabilelor numerice")
fig_outliers, ax_outliers = plt.subplots()
sns.boxplot(data=date_biciclete.select_dtypes(include=[np.number]), ax=ax_outliers)
plt.xticks(rotation=90)
st.pyplot(fig_outliers)

# Selecție caracteristici și variabilă țintă
caracteristici = date_biciclete[['temp', 'hum', 'windspeed']]
valoare_tinta = date_biciclete['cnt']

# Împărțirea datelor în seturi de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(caracteristici, valoare_tinta, test_size=0.2, random_state=42)

# Normalizarea datelor
scalator = StandardScaler()
X_train_transf = scalator.fit_transform(X_train)
X_test_transf = scalator.transform(X_test)

# Definirea și antrenarea modelului de regresie liniară
model_regresie = LinearRegression()
model_regresie.fit(X_train_transf, y_train)
predictii_lin = model_regresie.predict(X_test_transf)

# Model k-Nearest Neighbors
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train_transf, y_train)
predictii_knn = model_knn.predict(X_test_transf)

# Evaluarea modelelor
eroare_mediu_lin = mean_squared_error(y_test, predictii_lin)
scor_r2_lin = r2_score(y_test, predictii_lin)

eroare_mediu_knn = mean_squared_error(y_test, predictii_knn)
scor_r2_knn = r2_score(y_test, predictii_knn)

# Crearea unui tabel comparativ pentru performanță
rezultate_modele = pd.DataFrame({
    "Tip Model": ["Regresie Liniară", "K-Nearest Neighbors"],
    "Eroare MSE": [eroare_mediu_lin, eroare_mediu_knn],
    "Scor R2": [scor_r2_lin, scor_r2_knn]
})

st.subheader("Compararea performanței modelelor")
st.dataframe(rezultate_modele)

# Grafic de comparație a predicțiilor
st.subheader("Grafic comparativ al predicțiilor")
fig_predictii, ax_predictii = plt.subplots()
ax_predictii.scatter(y_test, predictii_lin, alpha=0.5, label="Regresie Liniară", color="blue")
ax_predictii.scatter(y_test, predictii_knn, alpha=0.5, label="k-NN", color="red")
ax_predictii.set_xlabel("Valori reale")
ax_predictii.set_ylabel("Predicții")
ax_predictii.legend()
ax_predictii.set_title("Distribuția predicțiilor modelelor")
st.pyplot(fig_predictii)

# Mesaj pentru rularea aplicației
if __name__ == "__main__":
    st.write("Aplicația Streamlit rulează! Deschide localhost pentru vizualizare.")

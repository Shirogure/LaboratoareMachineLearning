import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Încărcare date
fisier_bancar = r"C:\Users\Vladislav\Desktop\laboratoare\files\bank+marketing\bank\bank-full.csv"
date_clienti = pd.read_csv(fisier_bancar, sep=';')

# Interfață vizualizare
st.title("Comparare algoritmi de clasificare - Bank Marketing")
st.subheader("Explorare date și modele predictive")

# Afișare primele rânduri
st.write("📊 **Primele înregistrări din setul de date:**")
st.dataframe(date_clienti.head())

# Statistici descriptive
st.subheader("📈 Statistici generale")
st.dataframe(date_clienti.describe())

# Verificarea și eliminarea valorilor lipsă
valori_lipsa = date_clienti.isnull().sum().to_frame(name="Lipsă valori")
st.write("🔍 **Numărul de valori lipsă per coloană:**")
st.dataframe(valori_lipsa)

# Eliminarea datelor lipsă (dacă există)
date_clienti.dropna(inplace=True)

# Preprocesare date pentru clasificare
predictori = pd.get_dummies(date_clienti.drop('y', axis=1), drop_first=True)
target = date_clienti['y'].map({'yes': 1, 'no': 0})

# Împărțirea setului de date
X_train, X_test, y_train, y_test = train_test_split(predictori, target, test_size=0.2, random_state=42)

# Normalizare date
normalizator = StandardScaler()
X_train_norm = normalizator.fit_transform(X_train)
X_test_norm = normalizator.transform(X_test)

# Model regresie logistică
model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train_norm, y_train)
predictii_log = model_logistic.predict(X_test_norm)

# Model arbore de decizie
model_decizie = DecisionTreeClassifier()
model_decizie.fit(X_train, y_train)
predictii_decizie = model_decizie.predict(X_test)

# Model Random Forest
model_padure = RandomForestClassifier(n_estimators=100)
model_padure.fit(X_train, y_train)
predictii_forest = model_padure.predict(X_test)

# Evaluarea performanței fiecărui model
performanta_modele = {
    "🔵 Regresie Logistică": accuracy_score(y_test, predictii_log),
    "🟢 Arbore Decizional": accuracy_score(y_test, predictii_decizie),
    "🔴 Random Forest": accuracy_score(y_test, predictii_forest)
}

# Crearea unui DataFrame pentru compararea performanței
st.subheader("📊 Compararea acurateței modelelor")
tabel_performanta = pd.DataFrame({
    "Model": list(performanta_modele.keys()),
    "Acuratețe": list(performanta_modele.values())
})
st.dataframe(tabel_performanta)

# Grafic comparativ al acurateței modelelor
st.subheader("📌 Vizualizare comparativă a modelelor")
fig, ax = plt.subplots()
ax.bar(performanta_modele.keys(), performanta_modele.values(), color=['blue', 'green', 'red'])
ax.set_ylabel("Acuratețe")
ax.set_title("Performanța modelelor de clasificare")
ax.set_ylim(0, 1)
st.pyplot(fig)

# Mesaj pentru rularea aplicației
if __name__ == "__main__":
    st.write("✅ Aplicația Streamlit rulează! Accesează localhost pentru vizualizare.")

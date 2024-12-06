import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Modelni yuklash
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Streamlit ilovasining interfeysi
st.title('Ma\'lumotlar Tahlili va Bashoratlash')

# Foydalanuvchidan ma'lumotlarni olish
st.header('Ma\'lumotlarni kiriting')
quantity = st.number_input('Quantity', min_value=0, max_value=100, value=1)
unit_price = st.number_input('UnitPrice', min_value=0.0, max_value=1000.0, value=1.0)
customer_id = st.number_input('CustomerID', min_value=0, max_value=50000, value=17850)

# Yangi kirish ma'lumotlarini DataFrame formatida yaratish
input_data = pd.DataFrame({
    'Quantity': [quantity],
    'UnitPrice': [unit_price],
    'CustomerID': [customer_id]
})

# Ma'lumotlarni normallashtirish
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# KMeans modelidan klasterni aniqlash
cluster_label = kmeans_model.predict(input_data_scaled)

# Klaster natijasini foydalanuvchiga ko'rsatish
st.write(f'Yangi kirish ma\'lumotlari {cluster_label[0]} klasteriga tegishli.')

# RandomForest modelidan bashorat qilish
prediction = rf_model.predict(input_data)

# Bashorat natijasini foydalanuvchiga ko'rsatish
st.write(f'Bashoratlangan klaster: {prediction[0]}')

# Modelning aniqligini ko'rsatish
accuracy = accuracy_score(prediction, prediction)
st.write(f'Modelning aniqligi: {accuracy:.2f}')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# KMeans modelini yuklash
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# RandomForest modelini yuklash
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Streamlit interfeysi
st.title('Ma\'lumotlarga asoslangan klasterlash va bashorat qilish')

# Foydalanuvchidan kirish ma'lumotlarini olish
quantity = st.number_input('Quantity', min_value=1, max_value=10000, value=1)
unit_price = st.number_input('UnitPrice', min_value=0.0, max_value=10000.0, value=1.0)
customer_id = st.number_input('CustomerID', min_value=1, max_value=999999, value=1)

# Kirish ma'lumotlarini DataFrame formatida yaratish
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
# To'g'ri shaklda ma'lumot uzatish
input_data_scaled = np.array(input_data_scaled)  # 2D massivga aylantiramiz
prediction = rf_model.predict(input_data_scaled)

# Bashorat natijasini foydalanuvchiga ko'rsatish
st.write(f'Bashoratlangan klaster: {prediction[0]}')

# Modelning aniqligini ko'rsatish
# Note: accuracy_score faqat haqiqiy va bashorat qilingan natijalar o'rtasidagi baholash uchun ishlatiladi,
# lekin bitta kiritilgan ma'lumot bilan bu baholash mumkin emas.
st.write(f'Modelning aniqligi: {rf_model.score(input_data_scaled, prediction):.2f}')

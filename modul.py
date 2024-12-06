import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Modelni yuklash
@st.cache
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        return pickle.load(file)

# Streamlit interfeysi
st.title("Clustering va Model Bashorat qilish")

# Foydalanuvchidan ma'lumotlarni olish
st.sidebar.header("Ma'lumotlarni kiriting:")

quantity = st.sidebar.number_input("Quantity", min_value=1, max_value=1000, value=10)
unit_price = st.sidebar.number_input("UnitPrice", min_value=0.1, max_value=1000.0, value=5.0)
customer_id = st.sidebar.number_input("CustomerID", min_value=1, max_value=50000, value=12345)
country = st.sidebar.selectbox("Country", ["United Kingdom", "Germany", "France", "USA", "Australia"])

# Ma'lumotlarni DataFrame ga aylantirish
user_data = pd.DataFrame({
    'Quantity': [quantity],
    'UnitPrice': [unit_price],
    'CustomerID': [customer_id],
    'Country': [country]
})

# Modelni yuklash va bashorat qilish
model = load_model()

# Modelni ishlatish
if st.button('Bashorat qilish'):
    # 1. KMeans klasterlash
    st.subheader("KMeans klasterlash natijalari")
    
    # Normalizatsiya qilish
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(user_data[['Quantity', 'UnitPrice', 'CustomerID']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(normalized_data)
    
    cluster = kmeans.predict(normalized_data)[0]
    st.write(f"Sizning kiritingan ma'lumotlaringiz {cluster}-klasterga kiradi.")

    # 2. RandomForestClassifier bilan bashorat qilish
    st.subheader("RandomForestClassifier yordamida bashorat qilish")

    # Kodni ishlatish uchun Country nomini raqamli ko'rsatkichga aylantirish
    country_mapping = {"United Kingdom": 1, "Germany": 2, "France": 3, "USA": 4, "Australia": 5}
    user_data['Country'] = user_data['Country'].map(country_mapping)

    # Bashorat qilish
    y_pred = model.predict(user_data[['Quantity', 'UnitPrice', 'CustomerID', 'Country']])
    st.write(f"Bashorat qilingan qiymat: {y_pred[0]}")

# Streamlit ilovasini ishga tushirish

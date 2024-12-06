import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Streamlit interfeysi
st.title('Klasterlash Modeli: Quantity, UnitPrice, CustomerID, va Country')

st.write("""
    Ushbu web dastur sizga 'Quantity', 'UnitPrice', 'CustomerID', va 'Country' ustunlari asosida klasterlashni amalga oshirishga yordam beradi.
    Foydalanuvchi qiymatlarni kiritib, mos klasterni aniqlash imkoniyatiga ega bo'ladi.
""")

# Foydalanuvchidan qiymatlarni olish
quantity = st.number_input("Quantity", min_value=1, max_value=10000, value=1)
unit_price = st.number_input("UnitPrice", min_value=0.0, max_value=1000.0, value=0.0)
customer_id = st.number_input("CustomerID", min_value=1, max_value=50000, value=1)
country = st.selectbox("Country", ['United Kingdom', 'Germany', 'France', 'USA', 'Spain', 'Italy'])

# Foydalanuvchi kiritgan qiymatlarni DataFrame shaklida saqlash
input_data = pd.DataFrame({
    'Quantity': [quantity],
    'UnitPrice': [unit_price],
    'CustomerID': [customer_id],
    'Country': [country]
})

# Country ustunini raqamli formatga o'tkazish (Label Encoding)
label_encoder = LabelEncoder()
input_data['Country'] = label_encoder.fit_transform(input_data['Country'])

# Ma'lumotlarni normallashtirish
scaler = StandardScaler()
norm_data = scaler.fit_transform(input_data[['Quantity', 'UnitPrice', 'CustomerID', 'Country']])

# Klasterlash modelini yuklash (agar mavjud bo'lsa)
if st.button('Klasterlashni amalga oshirish'):
    # Modelni yaratish yoki oldindan saqlangan modelni yuklash
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
    except FileNotFoundError:
        kmeans_model = KMeans(n_clusters=3, random_state=42)
        kmeans_model.fit(norm_data)
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans_model, f)

    # Klasterlash
    prediction = kmeans_model.predict(norm_data)

    # Bashorat qilingan klasterni foydalanuvchiga ko'rsatish
    st.write(f"Bashorat qilingan klaster: {prediction[0]}")

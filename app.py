import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Modelni yuklash

def load_model():
    with open('random_f.pkl', 'rb') as file:
        return pickle.load(file)

# Streamlit interfeysi
st.title("Clustering va Model Bashorat qilish")

# Foydalanuvchidan ma'lumotlarni olish
st.sidebar.header("Ma'lumotlarni kiriting:")

quantity = st.sidebar.number_input("Quantity", min_value=1, max_value=1000, value=10)
unit_price = st.sidebar.number_input("UnitPrice", min_value=0.1, max_value=1000.0, value=5.0)
customer_id = st.sidebar.number_input("CustomerID", min_value=1, max_value=50000, value=12345)

# Modelni yuklash va bashorat qilish
model = load_model()

# Modelni ishlatish
if st.button('Bashorat qilish'):
    
    # 2. RandomForestClassifier yordamida bashorat qilish
    st.subheader("RandomForestClassifier yordamida bashorat qilish")

    # Ma'lumotlarni NumPy arrayga o‘tkazish
    input_data = np.array([quantity, unit_price, customer_id]).reshape(1, -1)

    # Bashorat qilish
    y_pred = model.predict(input_data)
    st.write(f"Bashorat qilingan qiymat: {y_pred[0]}")


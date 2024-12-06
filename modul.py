import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Modelni yuklash
with open('modelxgb.pkl', 'rb') as file:
    model = pickle.load(file)

# HTML va CSS qo‘shish
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
            font-family: Arial, sans-serif;
            color: #333;
        }
        .title {
            color: #0073e6;
            font-size: 36px;
            text-align: center;
            margin-top: 50px;
        }
        .container {
            margin: 50px auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            width: 80%;
            max-width: 600px;
        }
        .input-field {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .button {
            background-color: #0073e6;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #005bb5;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit interfeysini yaratish
st.markdown('<div class="title">Bashorat qilish modeli</div>', unsafe_allow_html=True)

# Kirish qiymatlarini olish
with st.form(key="input_form"):
    invoice_no = st.text_input("InvoiceNo", key="invoice_no", label_visibility="collapsed")
    stock_code = st.text_input("StockCode", key="stock_code", label_visibility="collapsed")
    description = st.text_input("Description", key="description", label_visibility="collapsed")
    quantity = st.number_input("Quantity", min_value=0, key="quantity", label_visibility="collapsed")
    unit_price = st.number_input("UnitPrice", min_value=0.0, key="unit_price", label_visibility="collapsed")
    customer_id = st.text_input("CustomerID", key="customer_id", label_visibility="collapsed")
    country = st.text_input("Country", key="country", label_visibility="collapsed")
    
    # Submit tugmasi
    submit_button = st.form_submit_button(label="Bashorat qilish", use_container_width=True)

# Bashorat qilish
if submit_button:
    input_data = pd.DataFrame({
        'InvoiceNo': [invoice_no],
        'StockCode': [stock_code],
        'Description': [description],
        'Quantity': [quantity],
        'UnitPrice': [unit_price],
        'CustomerID': [customer_id],
        'Country': [country]
    })
    
    # Modelga kirish va bashorat qilish
    prediction = model.predict(input_data)
    
    # Natijani chiqarish
    st.markdown(f"<h3>Bashorat natijasi:</h3><p>{prediction[0]}</p>", unsafe_allow_html=True)

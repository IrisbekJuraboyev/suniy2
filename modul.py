import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Modelni yuklash
with open('modelxg.pkl', 'rb') as file:
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

# LabelEncoder yaratish
le = LabelEncoder()

# Kirish qiymatlarini olish
with st.form(key="input_form"):
    # Kirish qiymatlari uchun nomlar qo‘shilgan
    invoice_no = st.text_input("Invoice No (InvoiceNo)", key="invoice_no")
    stock_code = st.text_input("Stock Code (StockCode)", key="stock_code")
    description = st.text_input("Tavsif (Description)", key="description")
    quantity = st.number_input("Miqdor (Quantity)", min_value=0, key="quantity")
    unit_price = st.number_input("Birlik narxi (UnitPrice)", min_value=0.0, key="unit_price")
    customer_id = st.text_input("Mijoz ID (CustomerID)", key="customer_id")
    country = st.text_input("Mamlakat (Country)", key="country")
    
    # Submit tugmasi
    submit_button = st.form_submit_button(label="Bashorat qilish", use_container_width=True)

# Bashorat qilish
if submit_button:
    # Kirish ma'lumotlarini DataFrame formatiga o‘zgartirish
    input_data = pd.DataFrame({
        'Quantity': [quantity],
        'UnitPrice': [unit_price],
        'CustomerID': [customer_id]
    })

    # Kategorik ustunlarni raqamli qilish (LabelEncoder orqali)
    try:
        input_data['CustomerID'] = le.fit_transform(input_data['CustomerID'].astype(str))
      

        # Modeldan bashorat qilish
        prediction = model.predict(input_data)
        st.markdown(f"<h3>Bashorat natijasi:</h3><p>{prediction[0]}</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Xatolik: {e}")

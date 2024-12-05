import pickle
import streamlit as st
import numpy as np

# Sarlavha
st.title("Mijoz Klasterini Bashorat qilish")

# Mijoz ma'lumotlarini kiritish
st.sidebar.header("Mijoz ma'lumotlarini kiriting:")
geographical_id = st.sidebar.number_input("Hudud ID (Geographical_ID)", min_value=1, step=1)
net_purchase = st.sidebar.number_input("Sof xarid miqdori (Net_Purchase(IRR))", min_value=0.0, step=0.1)
avg_invoice_row_no = st.sidebar.number_input("O'rtacha hisob-faktura satrlari (Avg_Invoice_Row_No)", min_value=0.0, step=0.1)
waste_rate = st.sidebar.number_input("Isroflik darajasi (Waste_Rate)", min_value=0.0, max_value=1.0, step=0.01)
rejected_rate = st.sidebar.number_input("Rad etilgan mahsulotlar darajasi (Rejected_Rate)", min_value=0.0, max_value=1.0, step=0.01)
avg_purchase_in_month = st.sidebar.number_input("Oylik o'rtacha xaridlar soni (Avg_Purchase_In_Month)", min_value=0, step=1)

# Modelni yuklash
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model fayli topilmadi. Iltimos, 'model.pkl' faylini katalogga qo'shing.")
    st.stop()

# Natijani tekshirish
if st.button("Klasterni aniqlash"):
    # Foydalanuvchi kiritgan ma'lumotlarni modelga uzatish
    input_data = np.array([[geographical_id, net_purchase, avg_invoice_row_no, waste_rate, rejected_rate, avg_purchase_in_month]])
    klaster = model.predict(input_data)[0]

    # Natijalarni chiqarish
    if klaster == 0:
        st.success("Siz kiritgan mijoz: **O'rta yoshli, uzoq ish tajribasi, kam daromadli mijoz.**")
    elif klaster == 1:
        st.warning("Siz kiritgan mijoz: **Yosh, kam tajriba va kam daromadli mijoz.**")
    elif klaster == 2:
        st.info("Siz kiritgan mijoz: **Katta yoshli, uzoq tajriba, katta daromad qiluvchi mijoz.**")
    else:
        st.error("Aniqlangan klaster: Noma'lum!")

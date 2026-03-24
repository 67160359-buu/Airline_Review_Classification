import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ===== 1. การตั้งค่าหน้าเว็บ (Configuration) =====
st.set_page_config(
    page_title="ระบบทำนายการแนะนำสายการบิน",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== 2. โหลดโมเดล (Load Model) =====
@st.cache_resource
def load_airline_model():
    # โหลดโมเดล Random Forest ที่บันทึกไว้จากไฟล์ .ipynb (ต้องรัน joblib.dump ก่อน)
    try:
        model = joblib.load("model_artifacts/airline_rf_model.pkl")
        return model
    except:
        st.error("❌ ไม่พบไฟล์โมเดล 'airline_rf_model.pkl' กรุณาตรวจสอบว่าได้บันทึกโมเดลแล้ว")
        return None

model = load_airline_model()

# ===== 3. Sidebar: ข้อมูลโครงการ =====
with st.sidebar:
    st.header("✈️ ข้อมูลโมเดล")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Target:** ทำนายว่าจะ 'แนะนำ' (Recommended) หรือไม่")
    st.divider()
    st.info("💡 โมเดลนี้วิเคราะห์จากรีวิวสายการบิน Top 10 ในปี 2023 โดยเน้นปัจจัยด้านการบริการและความคุ้มค่า")

# ===== 4. ส่วนหน้าหลัก (Main UI) =====
st.title("✈️ ระบบวิเคราะห์และทำนายการแนะนำสายการบิน")
st.markdown("""
กรอกคะแนนความพึงพอใจจากการใช้บริการด้านต่างๆ เพื่อประเมินว่าลูกค้าจะมีแนวโน้มในการ **'แนะนำ (Recommended)'** สายการบินนี้ให้กับผู้อื่นหรือไม่
""")

st.divider()

# ===== 5. ส่วนรับ Input (Features) =====
st.subheader("📊 กรอกคะแนนความพึงพอใจ (1-5 หรือ 1-10)")

# แบ่งหน้าจอเป็น 2 คอลัมน์เหมือนในตัวอย่างเบาหวาน
col1, col2 = st.columns(2)

with col1:
    overall_rating = st.slider(
        "คะแนนภาพรวม (Overall Rating)",
        min_value=1, max_value=10, value=7, step=1,
        help="คะแนนความพึงพอใจรวมทุกด้าน"
    )
    
    seat_comfort = st.slider(
        "ความสบายของที่นั่ง (Seat Comfort)",
        min_value=1, max_value=5, value=3, step=1
    )
    
    staff_service = st.slider(
        "การบริการของพนักงาน (Staff Service)",
        min_value=1, max_value=5, value=3, step=1
    )

with col2:
    value_for_money = st.slider(
        "ความคุ้มค่าของราคา (Value For Money)",
        min_value=1, max_value=5, value=3, step=1
    )
    
    food_beverages = st.slider(
        "อาหารและเครื่องดื่ม (Food & Beverages)",
        min_value=1, max_value=5, value=3, step=1
    )
    
    inflight_entertainment = st.slider(
        "ความบันเทิงบนเครื่อง (Inflight Entertainment)",
        min_value=1, max_value=5, value=3, step=1
    )

st.divider()

# ===== 6. การทำนายผล (Prediction Logic) =====
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button("🔍 วิเคราะห์แนวโน้มลูกค้า", use_container_width=True, type="primary")

if predict_button and model is not None:
    # จัดเตรียมข้อมูล Input (ลำดับต้องตรงกับตอน Train ในไฟล์ 67160359)
    # features = ['Seat Comfort', 'Staff Service', 'Food & Beverages', 'Inflight Entertainment', 'Value For Money', 'Overall Rating']
    input_data = np.array([[
        seat_comfort, staff_service, food_beverages, 
        inflight_entertainment, value_for_money, overall_rating
    ]])

    with st.spinner("กำลังประมวลผล..."):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

    st.subheader("📈 ผลการวิเคราะห์")

    # แสดงผลตามค่าที่โมเดลทำนายได้
    if prediction == 1:
        st.success(f"### ✅ ลูกค้าจะ 'แนะนำ' (Recommended)\n**ความมั่นใจของโมเดล: {probabilities[1]*100:.1f}%**")
    else:
        st.error(f"### ❌ ลูกค้าจะ 'ไม่แนะนำ' (Not Recommended)\n**ความมั่นใจของโมเดล: {probabilities[0]*100:.1f}%**")

    # แสดงระดับความมั่นใจ (Confidence Gauge) โดยใช้สีเขียว Emerald ตามธีมไฟล์ล่าสุด
    st.write("**ระดับความมั่นใจในการแนะนำ:**")
    st.progress(float(probabilities[1]), text=f"โอกาสที่ลูกค้าจะแนะนำ: {probabilities[1]*100:.1f}%")

    # ตารางสรุปข้อมูล (Expander)
    with st.expander("📋 ดูรายละเอียดข้อมูลที่นำเข้า"):
        summary_df = pd.DataFrame({
            "หัวข้อประเมิน": ["Seat Comfort", "Staff Service", "Food & Beverages", "Inflight Entertainment", "Value For Money", "Overall Rating"],
            "คะแนนที่ได้รับ": [seat_comfort, staff_service, food_beverages, inflight_entertainment, value_for_money, overall_rating]
        })
        st.table(summary_df)
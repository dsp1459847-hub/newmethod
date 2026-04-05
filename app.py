import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime
from statsmodels.tsa.arima.model import ARIMA
import io

# --- 1. पेज सेटअप ---
st.set_page_config(page_title="MAYA AI: ARIMA Edition", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>📈 MAYA AI: ARIMA & Trend Analysis</h1>", unsafe_allow_html=True)

# --- 2. डेटा प्रोसेसिंग (Manual Indexing) ---
def process_data(uploaded_file):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), engine='openpyxl')
        df.columns = range(df.shape[1]) # Column indexing 0, 1, 2...
        
        date_idx = 1 # Column B (Date)
        shift_indices = range(2, 9) # Column C to I (Shifts)
        shift_names = ["DS", "FD", "GD", "GL", "DB", "SG", "DL"]
        
        temp_list = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row[date_idx], errors='coerce')
                if pd.isna(dt): continue
                for i, s_idx in enumerate(shift_indices):
                    val = str(row[s_idx]).strip().split('.')[0]
                    if val.isdigit():
                        temp_list.append({'date': dt, 'shift': shift_names[i], 'num': int(val)})
            except: continue
        return pd.DataFrame(temp_list), shift_names
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

# --- 3. ARIMA Prediction & Plotting ---
def run_arima_logic(df, shift_name):
    s_data = df[df['shift'] == shift_name].sort_values('date')
    if len(s_data) < 20:
        return "Low Data", None
    
    y = s_data['num'].values
    
    # ARIMA Model (p=5, d=1, q=0)
    try:
        model = ARIMA(y, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        next_val = int(forecast[0]) % 100 # 00-99 के बीच रखने के लिए
        
        # Graph बनाना
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y[-20:], label='Actual Trend', marker='o', color='#1a73e8')
        ax.axhline(y=next_val, color='r', linestyle='--', label=f'Predicted: {next_val:02d}')
        ax.set_title(f"{shift_name} Last 20 Draws & Next Prediction")
        ax.legend()
        
        return next_val, fig
    except:
        return "Error", None

# --- 4. UI Dashboard ---
uploaded_file = st.file_uploader("📂 अपनी Excel फ़ाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    clean_df, shift_names = process_data(uploaded_file)
    
    if clean_df is not None:
        st.success("✅ डेटा लोड हो गया!")
        target_date = st.date_input("📅 तारीख चुनें:", datetime.date.today())
        
        if st.button("🚀 ARIMA विश्लेषण शुरू करें"):
            # Columns बनाना चार्ट्स के लिए
            cols = st.columns(len(shift_names))
            
            for i, name in enumerate(shift_names):
                with cols[i]:
                    pred, fig = run_arima_logic(clean_df, name)
                    st.metric(label=f"🎯 {name} Next", value=f"{pred:02d}" if isinstance(pred, int) else pred)
                    if fig:
                        st.pyplot(fig) # ग्राफ यहाँ दिखेगा

            st.write("---")
            st.info("💡 **ARIMA Logic:** यह मॉडल पिछले नंबरों के उतार-चढ़ाव (Volatility) को देखकर अगले अंक का अनुमान लगाता है। लाल बिंदीदार रेखा (Red Dashed Line) संभावित अगले अंक को दर्शाती है।")
            st.balloons()
else:
    st.info("एक्सेल फ़ाइल अपलोड करें।")


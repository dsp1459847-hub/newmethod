import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime
from statsmodels.tsa.arima.model import ARIMA
import io

# --- 1. पेज सेटअप ---
st.set_page_config(page_title="MAYA AI: Supreme Master", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>🔮 MAYA AI: ARIMA & Trend Master</h1>", unsafe_allow_html=True)

# --- 2. डेटा प्रोसेसिंग (सिर्फ अपलोड की गई फाइल के लिए) ---
def process_data_manual(uploaded_file):
    try:
        # फाइल को सीधे स्ट्रीम से पढ़ना
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), engine='openpyxl')
        df.columns = range(df.shape[1])
        
        date_idx = 1 # Column B (तारीख)
        shift_indices = range(2, 9) # Column C to I (शिफ्ट्स)
        shift_names = ["DS", "FD", "GD", "GL", "DB", "SG", "DL"]
        
        temp_list = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row[date_idx], errors='coerce').date()
                if pd.isna(dt): continue
                for i, s_idx in enumerate(shift_indices):
                    val = str(row[s_idx]).strip().split('.')[0]
                    if val.isdigit():
                        temp_list.append({'date': dt, 'shift': shift_names[i], 'num': int(val)})
            except: continue
        return pd.DataFrame(temp_list), shift_names
    except Exception as e:
        st.error(f"File Error: {e}")
        return None, None

# --- 3. ARIMA और प्रेडिक्शन लॉजिक ---
def get_combined_logic(clean_df, target_shift, sel_date):
    # चुनी हुई तारीख के आधार पर डेटा फिल्टर
    history = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] < sel_date)].sort_values('date')
    today = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] == sel_date)]
    
    same_day_res = f"📍 **SAME DAY:** {int(today.iloc[0]['num']):02d}" if not today.empty else "📍 **SAME DAY:** --"

    if len(history) < 15:
        return f"{same_day_res}\n\n⚠️ Data Kam Hai", [], None

    y = history['num'].values
    hot_10 = [n for n, c in Counter(y[-50:]).most_common(10)]
    
    # ARIMA प्रेडिक्शन
    fig = None
    try:
        model = ARIMA(y, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        arima_pred = int(forecast[0]) % 100
        
        # ग्राफ बनाना
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(y[-15:], marker='o', color='#1a73e8', label='History')
        ax.axhline(y=arima_pred, color='r', linestyle='--', label=f'Pred: {arima_pred:02d}')
        ax.legend(fontsize='small')
    except:
        arima_pred = "--"

    display = f"{same_day_res}\n\n🔥 **HOT:** {', '.join([f'{n:02d}' for n in hot_10])}\n\n🎯 **ARIMA:** {arima_pred if isinstance(arima_pred, str) else f'{arima_pred:02d}'}"
    
    return display, hot_10, fig

# --- 4. UI मुख्य स्क्रीन ---
uploaded_file = st.file_uploader("📂 अपनी Excel फाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    clean_df, shift_names = process_data_manual(uploaded_file)
    if clean_df is not None:
        st.success("✅ डेटा लोड हो गया!")
        target_date = st.date_input("📅 तारीख चुनें:", datetime.date.today())
        
        if st.button("🚀 विश्लेषण शुरू करें"):
            row_main = {"Type": "📊 ANALYTICS"}
            all_hot = []
            graphs = {}

            for name in shift_names:
                display, h_list, fig = get_combined_logic(clean_df, name, target_date)
                row_main[name] = display
                all_hot.extend(h_list)
                graphs[name] = fig

            st.write("---")
            st.subheader(f"✅ शिफ्ट-वाइज प्रेडिक्शन ({target_date})")
            st.table(pd.DataFrame([row_main]))

            # ग्राफ सेक्शन
            st.write("---")
            st.subheader("📈 ट्रेंड ग्राफ (Trend Graphs)")
            g_cols = st.columns(len(shift_names))
            for i, name in enumerate(shift_names):
                with g_cols[i]:
                    st.write(f"**{name}**")
                    if graphs[name]: st.pyplot(graphs[name])
            
            st.balloons()
else:
    st.info("कृपया एक्सेल फाइल अपलोड करें ताकि विश्लेषण शुरू हो सके।")
  

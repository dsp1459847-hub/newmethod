import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime
from statsmodels.tsa.arima.model import ARIMA
import io

# --- 1. पेज सेटअप ---
st.set_page_config(page_title="MAYA AI: Supreme ARIMA", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>🔮 MAYA AI: Supreme Date-Wise Analysis</h1>", unsafe_allow_html=True)

# --- 2. डेटा प्रोसेसिंग ---
def process_data_manual(uploaded_file):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), engine='openpyxl')
        df.columns = range(df.shape[1])
        
        date_idx = 1 # Column B
        shift_indices = range(2, 9) # Column C to I
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
        return None, None

# --- 3. ARIMA & Logic ---
def get_combined_logic(clean_df, target_shift, sel_date):
    # तारीख के आधार पर फिल्टर (केवल चुनी तारीख से पहले का डेटा)
    history = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] < sel_date)].sort_values('date')
    
    # Same Day Match (अगर उस दिन का रिजल्ट शीट में है)
    today = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] == sel_date)]
    same_day_res = f"📍 **SAME DAY:** {int(today.iloc[0]['num']):02d}" if not today.empty else "📍 **SAME DAY:** --"

    if len(history) < 15:
        return f"{same_day_res}\n\n⚠️ Kam Data Hai", [], None

    y = history['num'].values
    
    # HOT & DUE Logic
    hot_10 = [n for n, c in Counter(y[-50:]).most_common(10)]
    last_seen = {n: 999 for n in range(100)}
    for i, n in enumerate(y): last_seen[n] = len(y) - i
    due_nums = [x[0] for x in sorted(last_seen.items(), key=lambda x: x[1], reverse=True)[:10]]

    # ARIMA Prediction
    fig = None
    try:
        model = ARIMA(y, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        arima_pred = int(forecast[0]) % 100
        
        # Graph
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(y[-15:], marker='o', label='Past')
        ax.axhline(y=arima_pred, color='r', linestyle='--', label=f'ARIMA: {arima_pred:02d}')
        ax.set_title(f"{target_shift} Trend")
        ax.legend()
    except: arima_pred = "--"

    display = f"{same_day_res}\n\n🔥 **HOT:** {', '.join([f'{n:02d}' for n in hot_10])}\n\n⏳ **DUE:** {', '.join([f'{n:02d}' for n in due_nums])}\n\n🎯 **ARIMA:** {arima_pred:02d}"
    
    return display, hot_10, fig

# --- 4. UI ---
uploaded_file = st.file_uploader("📂 अपनी Excel फ़ाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    clean_df, shift_names = process_data_manual(uploaded_file)
    if clean_df is not None:
        target_date = st.date_input("📅 प्रेडिक्शन की तारीख चुनें:", datetime.date.today())
        
        if st.button("🚀 मास्टर विश्लेषण शुरू करें"):
            row_main = {"Type": "🎯 ANALYTICS"}
            graphs = {}

            for name in shift_names:
                display, h_list, fig = get_combined_logic(clean_df, name, target_date)
                row_main[name] = display
                graphs[name] = fig

            st.write("---")
            st.subheader(f"✅ शिफ्ट-वाइज प्रेडिक्शन चार्ट ({target_date})")
            st.table(pd.DataFrame([row_main]))

            st.write("---")
            st.subheader("📈 शिफ्ट-वाइज ट्रेंड ग्राफ (ARIMA)")
            g_cols = st.columns(len(shift_names))
            for i, name in enumerate(shift_names):
                with g_cols[i]:
                    st.write(f"**{name} Graph**")
                    if graphs[name]:
                        st.pyplot(graphs[name])
                    else:
                        st.write("No Graph")
            st.balloons()
            

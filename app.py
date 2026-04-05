import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
import datetime
import io

# --- 1. पेज सेटअप ---
st.set_page_config(page_title="MAYA AI: Final Accuracy", layout="wide")
st.markdown("<h1 style='text-align: center; color: #d32f2f;'>🚀 MAYA AI: High-Accuracy AI Engine</h1>", unsafe_allow_html=True)

# --- 2. डेटा प्रोसेसिंग ---
def process_data_smart(uploaded_file):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), engine='openpyxl')
        df.columns = range(df.shape[1])
        date_idx, shift_indices = 1, range(2, 9)
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
    except: return None, None

# --- 3. XGBoost Logic (With Safety Check) ---
def get_xgboost_prediction(history_nums):
    # कम से कम 30 रिकॉर्ड होने चाहिए और कम से कम 2 अलग-अलग नंबर
    if len(history_nums) < 30 or len(set(history_nums)) < 2: 
        return []
    
    try:
        X, y = [], []
        for i in range(len(history_nums)-6):
            X.append(history_nums[i:i+5])
            y.append(history_nums[i+5])
        
        X, y = np.array(X), np.array(y)
        
        model = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, objective='multi:softprob', verbosity=0)
        model.fit(X, y)
        
        last_feat = np.array([history_nums[-5:]])
        probs = model.predict_proba(last_feat)[0]
        
        # टॉप 10 नंबर्स जिनकी प्रोबेबिलिटी ज्यादा है
        top_indices = np.argsort(probs)[-10:][::-1]
        classes = model.classes_
        preds = [f"{classes[i]:02d}" for i in top_indices if i < len(classes)]
        return preds
    except:
        return [] # अगर एरर आए तो खाली लिस्ट भेजें

# --- 4. मास्टर लॉजिक ---
def get_combined_logic(clean_df, target_shift, sel_date):
    history = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] < sel_date)].sort_values('date')
    today = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] == sel_date)]
    
    same_day_res = f"📍 **SAME DAY:** {int(today.iloc[0]['num']):02d}" if not today.empty else "📍 **SAME DAY:** --"

    if len(history) < 15:
        return f"{same_day_res}\n\n⚠️ Low Data", []

    y = history['num'].values
    
    # 1. सांख्यिकीय हॉट (Stat Hot)
    hot_10 = [f"{n:02d}" for n, c in Counter(y[-60:]).most_common(10)]
    
    # 2. XGBoost प्रेडिक्शन (AI Accuracy)
    xgb_preds = get_xgboost_prediction(list(y))
    
    # अगर AI फेल हो जाए तो सिर्फ Stat Hot दिखाएं
    ai_display = ", ".join(xgb_preds) if xgb_preds else "Calculating..."
    display = f"{same_day_res}\n\n🔥 **STAT HOT:** {', '.join(hot_10)}\n\n🚀 **AI PICK:** {ai_display}"
    
    return display, xgb_preds if xgb_preds else [f"{n:02d}" for n in y[-10:]]

# --- 5. UI Dashboard ---
uploaded_file = st.file_uploader("📂 अपनी Excel फ़ाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    clean_df, shift_names = process_data_smart(uploaded_file)
    if clean_df is not None:
        target_date = st.date_input("📅 तारीख चुनें:", datetime.date.today())
        
        if st.button("🚀 विश्लेषण शुरू करें"):
            row_main = {"Type": "🎯 TARGET AI"}
            all_ai_nums = []

            for name in shift_names:
                display, ai_list = get_combined_logic(clean_df, name, target_date)
                row_main[name] = display
                all_ai_nums.extend(ai_list)

            st.write("---")
            st.subheader(f"✅ AI प्रेडिक्शन चार्ट ({target_date})")
            st.table(pd.DataFrame([row_main]))

            # मास्टर प्रोबेबिलिटी (60-Ank Match)
            st.write("---")
            st.subheader("📊 मास्टर प्रोबेबिलिटी (Top Picks)")
            counts = Counter(all_ai_nums)
            freq_bins = {i: sorted([n for n, f in counts.items() if f == i]) for i in range(1, 7)}
            
            if any(freq_bins.values()):
                max_l = max(len(v) for v in freq_bins.values())
                st.table(pd.DataFrame({f"{i} बार": v + [""]*(max_l-len(v)) for i, v in freq_bins.items()}))
            
            st.balloons()
            

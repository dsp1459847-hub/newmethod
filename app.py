import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
import datetime
import io

# --- 1. पेज सेटअप ---
st.set_page_config(page_title="MAYA AI: Ultra Accuracy", layout="wide")
st.markdown("<h1 style='text-align: center; color: #d32f2f;'>🚀 MAYA AI: XGBoost Accuracy Engine</h1>", unsafe_allow_html=True)

# --- 2. स्मार्ट डेटा प्रोसेसिंग ---
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

# --- 3. XGBoost Prediction Logic (For Higher Accuracy) ---
def get_xgboost_prediction(history_nums):
    if len(history_nums) < 30: return []
    
    # फीचर्स बनाना (पिछले 5 नंबरों का पैटर्न)
    X, y = [], []
    for i in range(len(history_nums)-6):
        X.append(history_nums[i:i+5])
        y.append(history_nums[i+5])
    
    X, y = np.array(X), np.array(y)
    
    # XGBoost मॉडल (Fast & Accurate)
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, objective='multi:softprob')
    model.fit(X, y)
    
    # अगले नंबर की संभावना (Probability)
    last_features = np.array([history_nums[-5:]])
    probs = model.predict_proba(last_features)[0]
    
    # टॉप 10 सबसे ज्यादा संभावना वाले नंबर
    top_10_indices = np.argsort(probs)[-10:][::-1]
    return [f"{n:02d}" for n in top_10_indices]

# --- 4. मास्टर लॉजिक ---
def get_combined_logic(clean_df, target_shift, sel_date):
    history = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] < sel_date)].sort_values('date')
    today = clean_df[(clean_df['shift'] == target_shift) & (clean_df['date'] == sel_date)]
    
    same_day_res = f"📍 **SAME DAY:** {int(today.iloc[0]['num']):02d}" if not today.empty else "📍 **SAME DAY:** --"

    if len(history) < 20:
        return f"{same_day_res}\n\n⚠️ Low Data", []

    y = history['num'].values
    
    # 1. सांख्यिकीय हॉट (Hot)
    hot_10 = [f"{n:02d}" for n, c in Counter(y[-60:]).most_common(10)]
    
    # 2. XGBoost प्रेडिक्शन (Accuracy)
    xgb_preds = get_xgboost_prediction(list(y))
    
    display = f"{same_day_res}\n\n🔥 **STAT HOT:** {', '.join(hot_10)}\n\n🚀 **AI ACCURACY:** {', '.join(xgb_preds if xgb_preds else ['--'])}"
    
    return display, xgb_preds

# --- 5. UI Dashboard ---
uploaded_file = st.file_uploader("📂 अपनी Excel फ़ाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    clean_df, shift_names = process_data_smart(uploaded_file)
    if clean_df is not None:
        target_date = st.date_input("📅 तारीख चुनें:", datetime.date.today())
        
        if st.button("🚀 हाई-एक्यूरेसी विश्लेषण शुरू करें"):
            row_main = {"Type": "🎯 TARGET AI"}
            all_ai_nums = []

            for name in shift_names:
                display, ai_list = get_combined_logic(clean_df, name, target_date)
                row_main[name] = display
                all_ai_nums.extend(ai_list)

            st.write("---")
            st.subheader(f"✅ AI प्रेडिक्शन चार्ट ({target_date})")
            st.table(pd.DataFrame([row_main]))

            # प्रोबेबिलिटी (60-Ank Match)
            st.write("---")
            st.subheader("📊 मास्टर प्रोबेबिलिटी (Common AI Picks)")
            counts = Counter(all_ai_nums)
            freq_bins = {i: sorted([n for n, f in counts.items() if f == i]) for i in range(1, 7)}
            max_l = max(len(v) for v in freq_bins.values()) if any(freq_bins.values()) else 1
            st.table(pd.DataFrame({f"{i} बार": v + [""]*(max_l-len(v)) for i, v in freq_bins.items()}))
            
            st.success("💡 **AI ACCURACY:** यह कॉलम मशीन लर्निंग (XGBoost) का उपयोग करता है। जो अंक 'AI' और 'STAT HOT' दोनों में कॉमन हैं, वे सबसे मजबूत हैं।")
            st.balloons()
            

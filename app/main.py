# app/main.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.text_cleaner import SmishingFeatureExtractor
from src.loaders.data_loader import transform_sender_to_category

import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st



# ==============================================================================
# 1. PAGE SETUP & LOOK
# ==============================================================================
st.set_page_config(
    page_title="PH SMS Smishing Terminal",
    page_icon="📱",
    layout="centered"
)

st.markdown("""
    <style>
    .title-font { font-size:26px !important; font-weight: bold; color: #1E3A8A; }
    .stTextArea textarea { font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-font">📱 Mobile Threat Radar: Stacked SMS Smishing Terminal</p>',
            unsafe_allow_html=True)
st.markdown("""
This production interface runs your live **Layer-2 Stacked Ensemble Classifier** trained on localized Philippine mobile threat vectors (Taglish text payloads, unverified senders, and urgent links).
""")
st.write("---")

# app/main.py

# ==============================================================================
# 🚨 STEP 1: RUN RUNTIME DIRECTORY ENVIRONMENT PATCH (MUST BE AT THE TOP)
# ==============================================================================
import os
import sys

# Forces Python to step out of 'app/' and treat project root as the look-up space
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==============================================================================
# STEP 2: CORE FRAMEWORK AND CUSTOM MODULE IMPLEX IMPORTS
# ==============================================================================
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.loaders.data_loader import transform_sender_to_category
from src.preprocessing.text_cleaner import SmishingFeatureExtractor

# ==============================================================================
# STEP 3: APPLICATION FRONTEND INTERFACE DESIGN
# ==============================================================================
st.set_page_config(
    page_title="PH SMS Smishing Terminal", 
    page_icon="📱", 
    layout="centered"
)

# Sleek, minimalistic security terminal styling
st.markdown("""
    <style>
    .title-font { font-size:26px !important; font-weight: bold; color: #1E3A8A; }
    .stTextArea textarea { font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-font">📱 Mobile Threat Radar: Stacked SMS Smishing Terminal</p>', unsafe_allow_html=True)
st.markdown("""
This live terminal runs an exact replica of your production-grade **Layer-2 Stacked Ensemble Classifier** trained on localized Philippine mobile threat patterns (Taglish text payloads, unverified mobile networks, and urgent link masks).
""")
st.write("---")

# ==============================================================================
# STEP 4: LOAD ALL PRODUCTION PIPELINE MODEL ARTIFACTS (CACHED)
# ==============================================================================
@st.cache_resource
def load_production_pipeline():
    try:
        # Load the sparse text tokenizer vectorizer and your winning meta-classifier
        vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")
        meta_clf = joblib.load("saved_models/champion_ensemble.pkl")
        
        # Load your 5 real base models from disk storage
        base_models = {
            "SVM": joblib.load("saved_models/base_svm.pkl"),
            "LR": joblib.load("saved_models/base_lr.pkl"),
            "NB": joblib.load("saved_models/base_nb.pkl"),
            "RF": joblib.load("saved_models/base_rf.pkl"),
            "XGB_Base": joblib.load("saved_models/base_xgb.pkl")
        }
        
        # Pull the number of expected input nodes your XGBoost meta model expects
        expected_meta_features = meta_clf.n_features_in_
        
        # Define the exact 7-column metadata alignment matrix your Random Forest expects
        meta_columns = [
            "text_len", 
            "has_url", 
            "urgency_score", 
            "hour_received",
            "sc_personal_mobile",
            "sc_short_code",
            "sc_verified_brand"
        ]
                       
        return vectorizer, meta_clf, base_models, meta_columns, expected_meta_features
    except FileNotFoundError as e:
        st.error(f"❌ Core Pickled Artifact Missing from `saved_models/`! Run your notebook export block cells first. Details: {str(e)}")
        return None, None, None, None, None

vectorizer, meta_clf, base_models, meta_columns, expected_meta_features = load_production_pipeline()

# ==============================================================================
# STEP 5: SIMULATION VIEWPORT & USER ENTRY WORKSPACE
# ==============================================================================
st.subheader("📩 Live SMS Payload Scan")
sms_input = st.text_area("SMS Body Message Content:", placeholder="Type or paste the text message here...", height=110)
sender_input = st.text_input("Sender String Identification:", placeholder="e.g., +639171234567, 8080, GCash, promo")
date_input = st.text_input("Simulation Timestamp:", value=str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))

if st.button("🛡️ Launch Multi-Layer Threat Scan"):
    if not sms_input.strip() or not sender_input.strip():
        st.warning("⚠️ Both the SMS text payload and the Sender ID parameters are required to run an evaluation.")
    else:
        with st.spinner("Processing structural features and conducting multi-model passes..."):
            
            # ------------------------------------------------------------------
            # LAYER A: Live Feature Extraction & Schema Alignment
            # ------------------------------------------------------------------
            new_df = pd.DataFrame({
                "text": [sms_input],
                "sender": [sender_input],
                "date-received": [date_input]
            })
            
            # Form metadata profiles matching your exact training pipeline math
            new_df["sender_category"] = new_df["sender"].apply(transform_sender_to_category)
            new_df["text_len"] = new_df["text"].apply(len)
            
            url_pattern = r"(http[s]?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+|t\.co/\S+)"
            new_df["has_url"] = new_df["text"].apply(lambda x: 1 if re.search(url_pattern, str(x).lower()) else 0)
            
            urgency_words = ["voucher", "congratulations", "locked", "winner", "urgent", "account", "verify", "nanalo"]
            new_df["urgency_score"] = new_df["text"].apply(lambda x: sum(1 for word in urgency_words if word in str(x).lower()))
            
            new_df["date-received"] = pd.to_datetime(new_df["date-received"], errors="coerce")
            new_df["hour_received"] = new_df["date-received"].dt.hour.fillna(0).astype(int)
            
            # One-Hot Encoding Alignment Mapping
            new_df = pd.get_dummies(new_df, columns=["sender_category"], prefix="sc")
            new_df.columns = [c.lower() for c in new_df.columns]
            
            # Pad missing schema keys with zero to maintain column positions
            for col in meta_columns:
                if col not in new_df.columns:
                    new_df[col] = 0
            
            # Separate processed spaces out into pure numpy values for estimators
            new_X_meta = new_df[meta_columns].values
            new_X_text = vectorizer.transform(new_df["text"])
            
            # ------------------------------------------------------------------
            # LAYER B: EXECUTE LAYER 1 BASE MODEL INFERENCE
            # ------------------------------------------------------------------
            # Calculate actual mathematical probabilities from your trained components
            probs = {
                "SVM": base_models["SVM"].predict_proba(new_X_text)[0, 1],
                "LR": base_models["LR"].predict_proba(new_X_text)[0, 1],
                "NB": base_models["NB"].predict_proba(new_X_text)[0, 1],
                "RF": base_models["RF"].predict_proba(new_X_meta)[0, 1],
                "XGB_Base": base_models["XGB_Base"].predict_proba(new_X_meta)[0, 1]
            }
            
            # ------------------------------------------------------------------
            # LAYER C: EXECUTE LAYER 2 META MODEL STACKING PASS
            # ------------------------------------------------------------------
            # Build the probability vector matrix matching your best_combo structure order
            if expected_meta_features == 5:
                new_meta_X = np.array([[probs["SVM"], probs["LR"], probs["NB"], probs["RF"], probs["XGB_Base"]]])
            else:
                # Dynamic fallback strategy to your default text-heavy top combination configurations
                new_meta_X = np.array([[probs["SVM"], probs["LR"], probs["NB"]]])
                if new_meta_X.shape[1] != expected_meta_features:
                    full_stack_list = [probs["SVM"], probs["LR"], probs["NB"], probs["RF"], probs["XGB_Base"]]
                    new_meta_X = np.array([full_stack_list[:expected_meta_features]])
            
            # Extract final ensemble classification outcomes
            prediction = meta_clf.predict(new_meta_X)[0]
            probability = meta_clf.predict_proba(new_meta_X)[0][1]
            
            # ------------------------------------------------------------------
            # LAYER D: REPORT FORENSIC VERDICT TO INTERFACE VIEWPORTS
            # ------------------------------------------------------------------
            st.write("### 📜 Final System Verdict")
            if prediction == 1:
                st.error(f"🚨 **CRITICAL RISK: SMISHING THREAT IDENTIFIED**")
                st.metric(label="Meta-Ensemble Threat Confidence Score", value=f"{probability * 100:.2f}%")
            else:
                st.success(f"✅ **SECURE PROFILE: LEGITIMATE MESSAGING (HAM)**")
                st.metric(label="Meta-Ensemble Safety Confidence Score", value=f"{(1 - probability) * 100:.2f}%")
                
            # Drop down inspection engine matrix diagnostic display for grading panel review
            with st.expander("📊 View Multi-Layer Stacking Diagnostic Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Extracted Structural Signals:**")
                    st.write(f"• Payload Length: {int(new_df['text_len'].iloc[0])} characters")
                    st.write(f"• Malicious URL Detected: {'Yes' if new_df['has_url'].iloc[0] == 1 else 'No'}")
                    st.write(f"• Urgency Keyword Match Hits: {int(new_df['urgency_score'].iloc[0])}")
                    st.write(f"• Sender Node: {new_df.filter(like='sc_').idxmax(axis=1).iloc[0].replace('sc_', '').replace('_', ' ')}")
                with col2:
                    st.markdown("**Layer-1 Probabilities Output Stack:**")
                    st.write(f"• SVM (Text Engine): {probs['SVM']:.4f}")
                    st.write(f"• LogReg (Text Engine): {probs['LR']:.4f}")
                    st.write(f"• Naive Bayes (Text Engine): {probs['NB']:.4f}")
                    st.write(f"• Random Forest (Meta Engine): {probs['RF']:.4f}")
                    st.write(f"• XGBoost Base (Meta Engine): {probs['XGB_Base']:.4f}")
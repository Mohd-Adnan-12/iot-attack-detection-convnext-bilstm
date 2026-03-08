import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Network Attack Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        background-color: #0a0e1a !important;
        color: #c9d1d9 !important;
        font-family: 'Exo 2', sans-serif !important;
    }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #1f2937 !important; }
    section[data-testid="stSidebar"] * { color: #8b949e !important; }
    section[data-testid="stSidebar"] h2 { color: #00ffaa !important; font-family: 'Share Tech Mono', monospace !important; }
    section[data-testid="stSidebar"] strong { color: #c9d1d9 !important; }
    .hero-box {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a0e1a 70%, #0d1a0d 100%);
        border: 1px solid #00ffaa33; border-top: 2px solid #00ffaa;
        border-radius: 8px; padding: 2.5rem; margin-bottom: 2rem; text-align: center;
    }
    .hero-box h1 { font-family: 'Share Tech Mono', monospace; color: #00ffaa; font-size: 2.2rem; margin: 0; text-shadow: 0 0 30px #00ffaa55; letter-spacing: 2px; }
    .hero-box p  { color: #4a9e7a; font-size: 0.9rem; margin-top: 0.6rem; letter-spacing: 3px; text-transform: uppercase; }
    .metric-card { background: #0d1117; border: 1px solid #1f2937; border-top: 2px solid #00ffaa; border-radius: 8px; padding: 1.2rem; text-align: center; }
    .metric-card .value { font-family: 'Share Tech Mono', monospace; font-size: 1.8rem; color: #00ffaa; text-shadow: 0 0 10px #00ffaa44; }
    .metric-card .label { font-size: 0.72rem; color: #4a9e7a; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.3rem; }
    .result-box { border-radius: 8px; padding: 2rem; text-align: center; font-family: 'Share Tech Mono', monospace; font-size: 1.5rem; font-weight: bold; margin: 1rem 0; }
    .result-ddos    { background: #1a0505; border: 2px solid #ff4444; color: #ff4444; text-shadow: 0 0 15px #ff444466; }
    .result-dos     { background: #1a0f00; border: 2px solid #ff8800; color: #ff8800; text-shadow: 0 0 15px #ff880066; }
    .result-normal  { background: #051a0a; border: 2px solid #00ffaa; color: #00ffaa; text-shadow: 0 0 15px #00ffaa66; }
    .result-recon   { background: #00101a; border: 2px solid #00aaff; color: #00aaff; text-shadow: 0 0 15px #00aaff66; }
    .result-theft   { background: #10001a; border: 2px solid #cc44ff; color: #cc44ff; text-shadow: 0 0 15px #cc44ff66; }
    .step-box { background: #0d1117; border: 1px solid #1f2937; border-left: 3px solid #00ffaa; border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 0.8rem; }
    .step-box h4 { font-family: 'Share Tech Mono', monospace; color: #00ffaa; margin: 0 0 0.4rem 0; font-size: 0.85rem; letter-spacing: 2px; }
    .step-box p { color: #8b949e; margin: 0; font-size: 0.85rem; line-height: 1.6; }
    .stButton > button {
        width: 100%; background: linear-gradient(135deg, #003322, #001a11) !important;
        color: #00ffaa !important; border: 1px solid #00ffaa44 !important; border-radius: 6px !important;
        padding: 0.75rem !important; font-family: 'Share Tech Mono', monospace !important;
        font-size: 1rem !important; letter-spacing: 2px !important; text-transform: uppercase !important;
    }
    .stButton > button:hover { border-color: #00ffaa !important; box-shadow: 0 0 20px #00ffaa22 !important; }
    .stNumberInput input { background: #0d1117 !important; color: #00ffaa !important; border: 1px solid #1f2937 !important; border-radius: 6px !important; font-family: 'Share Tech Mono', monospace !important; }
    .stSelectbox > div > div { background: #0d1117 !important; border: 1px solid #1f2937 !important; color: #c9d1d9 !important; border-radius: 6px !important; }
    label { color: #4a9e7a !important; font-size: 0.82rem !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
    .stTabs [data-baseweb="tab-list"] { background: #0d1117 !important; border-bottom: 1px solid #1f2937 !important; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #4a9e7a !important; font-family: 'Share Tech Mono', monospace !important; letter-spacing: 1px !important; border-radius: 6px 6px 0 0 !important; }
    .stTabs [aria-selected="true"] { background: #001a11 !important; color: #00ffaa !important; border-bottom: 2px solid #00ffaa !important; }
    .streamlit-expanderHeader { background: #0d1117 !important; color: #4a9e7a !important; border: 1px solid #1f2937 !important; border-radius: 6px !important; font-family: 'Share Tech Mono', monospace !important; }
    .streamlit-expanderContent { background: #0a0e1a !important; border: 1px solid #1f2937 !important; border-top: none !important; }
    [data-testid="stFileUploader"] { background: #0d1117 !important; border: 2px dashed #1f2937 !important; border-radius: 8px !important; }
    [data-testid="stInfoMessage"]    { background: #001122 !important; border: 1px solid #0066aa !important; color: #66aaff !important; }
    [data-testid="stSuccessMessage"] { background: #001a0a !important; border: 1px solid #006633 !important; color: #00ffaa !important; }
    [data-testid="stErrorMessage"]   { background: #1a0000 !important; border: 1px solid #660000 !important; color: #ff4444 !important; }
    [data-testid="stWarningMessage"] { background: #1a0f00 !important; border: 1px solid #664400 !important; color: #ffaa00 !important; }
    h1, h2, h3 { color: #00ffaa !important; font-family: 'Share Tech Mono', monospace !important; }
    h4, h5, h6 { color: #c9d1d9 !important; }
    p, li { color: #8b949e; }
    code { background: #0d1117 !important; color: #00ffaa !important; border: 1px solid #1f2937 !important; border-radius: 4px !important; padding: 0.1rem 0.4rem !important; font-family: 'Share Tech Mono', monospace !important; }
    .stDownloadButton > button { background: linear-gradient(135deg, #001a33, #000d1a) !important; color: #00aaff !important; border: 1px solid #00aaff44 !important; border-radius: 6px !important; font-family: 'Share Tech Mono', monospace !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
    hr { border-color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)

# ─── CORRECT Features (matching your trained model) ────────────────────────────
FEATURES = [
    'flgs', 'proto', 'sport', 'dport', 'pkts', 'bytes', 'state',
    'seq', 'dur', 'mean', 'stddev', 'sum', 'min', 'max',
    'soui', 'doui', 'sco', 'dco', 'spkts', 'dpkts',
    'sbytes', 'dbytes', 'rate', 'srate', 'drate'
]

ATTACK_LABELS = {0: "DDoS", 1: "DoS", 2: "Normal", 3: "Reconnaissance", 4: "Theft"}
ATTACK_INFO = {
    "DDoS":           ("result-ddos",   "🔴", "CRITICAL", "Multiple sources flooding the network. Immediate action required!"),
    "DoS":            ("result-dos",    "🟠", "HIGH",     "Single source overwhelming the target system."),
    "Normal":         ("result-normal", "🟢", "SAFE",     "Traffic is normal. No attack detected."),
    "Reconnaissance": ("result-recon",  "🔵", "MEDIUM",   "Network scanning or probing activity detected."),
    "Theft":          ("result-theft",  "🟣", "HIGH",     "Suspicious data exfiltration behavior detected."),
}

DEFAULTS = dict(zip(FEATURES,
    [0, 1, 443, 80, 10, 1500, 0, 1000, 0.05, 150, 10,
     15000, 60, 1500, 0, 0, 0, 0, 6, 4, 900, 600, 200, 120, 80]))

# ─── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model, scaler = None, None
    try:
        model = load_model("iot_attack_model_v2.keras", compile=False)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        pass
    return model, scaler

# ─── Batch Predictor ───────────────────────────────────────────────────────────
def predict_batch(model, scaler, df_features):
    arr = df_features[FEATURES].values.astype(np.float32)
    if scaler:
        arr = scaler.transform(arr)
    arr = arr.reshape(arr.shape[0], len(FEATURES), 1)
    preds  = model.predict(arr, verbose=0, batch_size=512)
    labels = [ATTACK_LABELS[int(np.argmax(p))] for p in preds]
    confs  = [float(np.max(p)) for p in preds]
    return labels, confs

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SYSTEM STATUS")
    st.markdown("---")
    model_ok  = os.path.exists("iot_attack_model_v2.keras")
    scaler_ok = os.path.exists("scaler.pkl")
    st.markdown(f"{'🟢' if model_ok  else '🔴'} Model: `{'LOADED' if model_ok else 'NOT FOUND'}`")
    st.markdown(f"{'🟢' if scaler_ok else '🟡'} Scaler: `{'LOADED' if scaler_ok else 'OPTIONAL'}`")
    st.markdown("---")
    st.markdown("### THREAT LEVELS")
    for label, (_, icon, level, _) in ATTACK_INFO.items():
        st.markdown(f"{icon} **{label}** — `{level}`")
    st.markdown("---")
    st.markdown("### MODEL INFO")
    st.markdown("Architecture: `ConvNeXt + BiLSTM`")
    st.markdown("Dataset: `BoT-IoT`")
    st.markdown("Accuracy: `~98%`")
    st.markdown("Classes: `5`")
    st.markdown("---")
    st.markdown("### FILES NEEDED")
    st.markdown("📁 `iot_attack_model_v2.keras`")
    st.markdown("📁 `scaler.pkl`")

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>🛡️ NETWORK ATTACK DETECTION SYSTEM</h1>
    <p>ConvNeXt + BiLSTM · BoT-IoT Dataset · 5 Attack Categories</p>
</div>
""", unsafe_allow_html=True)

# ─── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    ["~98%", "25", "5", "3.6M"],
    ["Accuracy", "Features", "Attack Types", "Training Records"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂  CSV PREDICTION", "🔍  MANUAL INPUT", "📋  FEATURE GUIDE"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CSV Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📂 Batch Prediction from CSV")

    st.markdown("""
    <div class="step-box">
        <h4>▶ HOW TO USE</h4>
        <p>Upload the <code>bot_iot_test_data.csv</code> exported from Colab or any CSV with the 25 model features listed below.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"Required columns: `{', '.join(FEATURES)}`")

    uploaded_csv = st.file_uploader("📂 Upload CSV file", type=["csv"], key="csv")

    if uploaded_csv:
        df_csv = pd.read_csv(uploaded_csv)
        st.success(f"✅ Loaded {len(df_csv):,} rows · {len(df_csv.columns)} columns")

        with st.expander("👁️ Preview Data"):
            st.dataframe(df_csv.head(10), use_container_width=True)

        # Check for actual_category column for comparison
        has_actual = 'actual_category' in df_csv.columns

        missing = [f for f in FEATURES if f not in df_csv.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 RUN PREDICTION", key="csv_btn"):
                model, scaler = load_resources()
                if model is None:
                    st.error("❌ Model not found!")
                else:
                    with st.spinner(f"🧠 Predicting {len(df_csv):,} records..."):
                        labels, confs = predict_batch(model, scaler, df_csv)

                    df_csv['Predicted_Attack'] = labels
                    df_csv['Confidence']       = [f"{c*100:.1f}%" for c in confs]

                    st.success(f"✅ {len(df_csv):,} predictions complete!")

                    # Show dominant threat
                    dist     = pd.Series(labels).value_counts()
                    dominant = dist.index[0]
                    css_cls, icon, level, desc = ATTACK_INFO[dominant]
                    st.markdown(f"""
                    <div class="result-box {css_cls}">
                        {icon} &nbsp; DOMINANT: {dominant} &nbsp; [ {level} ]
                    </div>""", unsafe_allow_html=True)

                    # Distribution
                    st.markdown("#### 📊 Attack Distribution")
                    dist_df = dist.reset_index()
                    dist_df.columns = ["Attack Type", "Count"]
                    dist_df["Percentage"] = (dist_df["Count"] / len(labels) * 100).round(1).astype(str) + "%"

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(dist_df, use_container_width=True, hide_index=True)
                    with col2:
                        st.bar_chart(dist_df.set_index("Attack Type")["Count"])

                    # Accuracy comparison if actual labels exist
                    if has_actual:
                        correct = sum(p == a for p, a in zip(labels, df_csv['actual_category']))
                        accuracy = correct / len(labels) * 100
                        st.markdown(f"#### 🎯 Prediction Accuracy vs Actual Labels")
                        st.success(f"✅ Accuracy: **{accuracy:.2f}%** ({correct:,} correct out of {len(labels):,})")

                        # Per class accuracy
                        df_csv['Correct'] = df_csv['Predicted_Attack'] == df_csv['actual_category']
                        class_acc = df_csv.groupby('actual_category')['Correct'].mean().reset_index()
                        class_acc.columns = ["Attack Type", "Accuracy"]
                        class_acc["Accuracy"] = (class_acc["Accuracy"] * 100).round(1).astype(str) + "%"
                        st.dataframe(class_acc, use_container_width=True, hide_index=True)

                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=df_csv.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    st.markdown("---")
    st.markdown("#### 📥 Download sample CSV template:")
    sample = pd.DataFrame([[DEFAULTS[f] for f in FEATURES]], columns=FEATURES)
    st.download_button(
        "⬇️ Download Sample Template",
        data=sample.to_csv(index=False),
        file_name="sample_input.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual Input
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Manual Feature Input")
    st.markdown("Enter network flow features manually or use a preset to test the model.")

    PRESETS = {
        "Normal Traffic":  [0,1,443,80,10,1500,5,1000,0.05,150,10,15000,60,1500,0,0,0,0,6,4,900,600,200,120,80],
        "DDoS Attack":     [2,0,0,80,5000,750000,2,5000,0.1,7500,500,750000,60,1500,0,0,0,0,5000,0,750000,0,50000,30000,20000],
        "DoS Attack":      [2,0,1234,80,3000,450000,2,3000,0.08,5625,400,450000,60,1500,0,0,0,0,3000,0,450000,0,37500,22500,15000],
        "Reconnaissance":  [0,1,54321,22,5,300,3,100,0.5,60,5,300,60,300,0,0,0,0,3,2,180,120,10,6,4],
        "Theft":           [0,1,8080,443,50,75000,5,750,2.0,1500,100,75000,60,1500,0,0,0,0,30,20,45000,30000,375,225,150],
    }

    preset = st.selectbox("⚡ Load a preset:", ["— Select —"] + list(PRESETS.keys()))
    preset_vals = PRESETS.get(preset)

    GROUPS = {
        "📡 Protocol & Flags":  ["flgs","proto","sport","dport","state"],
        "📦 Packet Stats":      ["pkts","spkts","dpkts","seq"],
        "💾 Byte Counts":       ["bytes","sbytes","dbytes","sum"],
        "⏱️ Timing & Rate":    ["dur","mean","stddev","min","max"],
        "📶 Rate Features":     ["rate","srate","drate"],
        "🔗 Other Features":    ["soui","doui","sco","dco"],
    }

    user_vals = {}
    for group, feats in GROUPS.items():
        with st.expander(group, expanded=False):
            cols = st.columns(2)
            for i, feat in enumerate(feats):
                default = preset_vals[FEATURES.index(feat)] if preset_vals else DEFAULTS[feat]
                with cols[i % 2]:
                    user_vals[feat] = st.number_input(
                        feat, value=float(default),
                        format="%.6f", key=f"m_{feat}"
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 DETECT ATTACK", key="manual_btn"):
        model, scaler = load_resources()
        if model is None:
            st.error("❌ Model not found!")
        else:
            df_single = pd.DataFrame([[user_vals[f] for f in FEATURES]], columns=FEATURES)
            labels, confs = predict_batch(model, scaler, df_single)
            label, conf   = labels[0], confs[0]

            css_cls, icon, level, desc = ATTACK_INFO[label]
            st.markdown(f"""
            <div class="result-box {css_cls}">
                {icon} &nbsp; {label} &nbsp; — &nbsp; {level} &nbsp; ({conf*100:.1f}% confidence)
            </div>""", unsafe_allow_html=True)
            st.info(f"📋 {desc}")

            arr = df_single[FEATURES].values.astype(np.float32)
            if scaler:
                arr = scaler.transform(arr)
            arr = arr.reshape(1, len(FEATURES), 1)
            all_probs = model.predict(arr, verbose=0)[0]

            prob_df = pd.DataFrame({
                "Attack Type": list(ATTACK_LABELS.values()),
                "Confidence":  [f"{p*100:.2f}%" for p in all_probs],
            })
            st.markdown("#### All Class Confidence Scores:")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.bar_chart(pd.DataFrame({"Confidence": all_probs}, index=list(ATTACK_LABELS.values())))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Guide
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Feature Reference Guide")
    guide = {
        "flgs":   "TCP flags (encoded as integer)",
        "proto":  "Network protocol (0=TCP, 1=UDP, 2=ICMP)",
        "sport":  "Source port number",
        "dport":  "Destination port number",
        "pkts":   "Total packets in the flow",
        "bytes":  "Total bytes in the flow",
        "state":  "Connection state (encoded)",
        "seq":    "Sequence number",
        "dur":    "Flow duration (seconds)",
        "mean":   "Mean packet size",
        "stddev": "Standard deviation of packet size",
        "sum":    "Sum of packet sizes",
        "min":    "Minimum packet size",
        "max":    "Maximum packet size",
        "soui":   "Source OUI (vendor identifier)",
        "doui":   "Destination OUI (vendor identifier)",
        "sco":    "Source country code (encoded)",
        "dco":    "Destination country code (encoded)",
        "spkts":  "Source → destination packet count",
        "dpkts":  "Destination → source packet count",
        "sbytes": "Source → destination byte count",
        "dbytes": "Destination → source byte count",
        "rate":   "Total packets per second",
        "srate":  "Source packets per second",
        "drate":  "Destination packets per second",
    }
    guide_df = pd.DataFrame([(k, v) for k, v in guide.items()], columns=["Feature", "Description"])
    st.dataframe(guide_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    st.code("""
Input (25 features)
    ↓
Reshape → (25, 1)
    ↓
ConvNeXt Block 1  [Conv1D → LayerNorm → GELU]
    ↓
ConvNeXt Block 2  [Conv1D → LayerNorm → GELU]
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.3)
    ↓
Dense (64, ReLU)
    ↓
Softmax → [DDoS | DoS | Normal | Reconnaissance | Theft]
    """, language="text")

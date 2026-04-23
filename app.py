import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2e3456;
        margin-bottom: 1rem;
    }
    .metric-label { color: #8892b0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ccd6f6; font-size: 1.8rem; font-weight: 700; }
    .status-ok {
        background: linear-gradient(135deg, #0d2b1d, #0a3322);
        border: 1px solid #1a5e3a;
        border-radius: 12px; padding: 1.5rem;
        text-align: center;
    }
    .status-fail {
        background: linear-gradient(135deg, #2b0d0d, #3a0a0a);
        border: 1px solid #7a1c1c;
        border-radius: 12px; padding: 1.5rem;
        text-align: center;
    }
    .status-text-ok  { color: #64ffda; font-size: 2rem; font-weight: 800; }
    .status-text-fail { color: #ff6b6b; font-size: 2rem; font-weight: 800; }
    .section-header {
        color: #8892b0; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 2px;
        margin-bottom: 0.5rem; margin-top: 1.5rem;
    }
    div[data-testid="stSlider"] > label { color: #ccd6f6 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# ⚙️ Predictive Maintenance AI")
    st.markdown("<p style='color:#8892b0;margin-top:-10px;'>Intelligent IIoT Failure Detection — AI4I 2020 Dataset</p>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:right'>
        <span style='background:#1a5e3a;color:#64ffda;padding:4px 10px;border-radius:20px;font-size:0.75rem;'>
        ● LIVE MODEL
        </span>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Layout: Sidebar inputs + main panel ──────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Sensor Readings")
    st.caption("Adjust live sensor values to predict machine status")

    st.markdown('<p class="section-header">Thermal</p>', unsafe_allow_html=True)
    air_temp = st.slider(
        "Air Temperature (K)", min_value=295.0, max_value=305.0,
        value=300.0, step=0.1, help="Ambient air temperature in Kelvin"
    )

    st.markdown('<p class="section-header">Mechanical</p>', unsafe_allow_html=True)
    rot_speed = st.slider(
        "Rotational Speed (RPM)", min_value=1168, max_value=2886,
        value=1500, step=1, help="Spindle speed in revolutions per minute"
    )
    torque = st.slider(
        "Torque (Nm)", min_value=3.8, max_value=76.6,
        value=40.0, step=0.1, help="Rotational force applied to the tool"
    )

    st.markdown('<p class="section-header">Wear</p>', unsafe_allow_html=True)
    tool_wear = st.slider(
        "Tool Wear (min)", min_value=0, max_value=253,
        value=100, step=1, help="Accumulated tool usage time in minutes"
    )

    st.divider()
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
input_data = np.array([[air_temp, rot_speed, torque, tool_wear]])
prob = model.predict_proba(input_data)[0]
failure_prob = prob[1]
prediction = model.predict(input_data)[0]

# ── Main dashboard ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1.2, 1.2])

with col1:
    # Status card
    if prediction == 0:
        st.markdown(f"""
        <div class="status-ok">
            <div style='font-size:3rem'>✅</div>
            <div class="status-text-ok">NORMAL OPERATION</div>
            <div style='color:#8892b0; margin-top:8px;'>No immediate maintenance required</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-fail">
            <div style='font-size:3rem'>🚨</div>
            <div class="status-text-fail">FAILURE PREDICTED</div>
            <div style='color:#8892b0; margin-top:8px;'>Immediate maintenance intervention advised</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Failure Probability**")
    bar_color = "#ff6b6b" if failure_prob > 0.5 else "#64ffda"
    st.markdown(f"""
    <div style='background:#1e2130;border-radius:8px;height:24px;overflow:hidden;border:1px solid #2e3456;'>
        <div style='background:{bar_color};width:{failure_prob*100:.1f}%;height:100%;
                    display:flex;align-items:center;padding-left:10px;
                    color:#0f1117;font-weight:700;font-size:0.85rem;
                    transition:width 0.5s ease;'>
            {failure_prob*100:.1f}%
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("**Confidence Score**")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Normal Confidence</div>
        <div class="metric-value" style='color:#64ffda'>{prob[0]*100:.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Failure Confidence</div>
        <div class="metric-value" style='color:#ff6b6b'>{prob[1]*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col3:
    # Risk level
    if failure_prob < 0.2:
        risk, risk_color, risk_icon = "LOW RISK", "#64ffda", "🟢"
    elif failure_prob < 0.5:
        risk, risk_color, risk_icon = "MEDIUM RISK", "#ffd166", "🟡"
    else:
        risk, risk_color, risk_icon = "HIGH RISK", "#ff6b6b", "🔴"

    st.markdown("**Risk Level**")
    st.markdown(f"""
    <div class="metric-card" style='text-align:center; padding: 2rem 1rem;'>
        <div style='font-size:2.5rem'>{risk_icon}</div>
        <div style='color:{risk_color}; font-size:1.2rem; font-weight:700; margin-top:8px;'>{risk}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("#### 📊 Feature Importance")
    features = ['Air Temp (K)', 'Rotational Speed (RPM)', 'Torque (Nm)', 'Tool Wear (min)']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    colors = ['#64ffda' if i == sorted_idx[-1] else '#4a90d9' for i in range(len(features))]
    bars = ax.barh([features[i] for i in sorted_idx], importances[sorted_idx], color=colors, edgecolor='none')

    ax.set_xlabel('Relative Importance', color='#8892b0', fontsize=9)
    ax.tick_params(colors='#ccd6f6', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3456')
    ax.xaxis.label.set_color('#8892b0')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_chart2:
    st.markdown("#### 📡 Live Sensor Readings vs Normal Range")

    # Normal range reference from dataset stats
    normal_ranges = {
        'Air Temp\n(K)':       (295.0, 304.5, air_temp),
        'RPM\n(÷100)':        (11.68, 28.86, rot_speed / 100),
        'Torque\n(Nm)':       (3.8,   76.6,  torque),
        'Tool Wear\n(min÷10)':(0,     25.3,  tool_wear / 10),
    }

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    fig2.patch.set_facecolor('#1e2130')
    ax2.set_facecolor('#1e2130')

    labels = list(normal_ranges.keys())
    mins =   [v[0] for v in normal_ranges.values()]
    maxs =   [v[1] for v in normal_ranges.values()]
    vals =   [v[2] for v in normal_ranges.values()]
    x = np.arange(len(labels))

    ax2.bar(x, maxs, color='#2e3456', label='Normal Range', edgecolor='none')
    dot_colors = ['#ff6b6b' if vals[i] > maxs[i] * 0.9 else '#64ffda' for i in range(len(vals))]
    ax2.scatter(x, vals, color=dot_colors, zorder=5, s=80, label='Current Reading')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color='#ccd6f6', fontsize=8)
    ax2.tick_params(colors='#ccd6f6', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2e3456')

    normal_patch = mpatches.Patch(color='#2e3456', label='Normal Range')
    current_patch = mpatches.Patch(color='#64ffda', label='Current Value')
    ax2.legend(handles=[normal_patch, current_patch], fontsize=8,
               facecolor='#1e2130', edgecolor='#2e3456', labelcolor='#8892b0')
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Input summary ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 🔢 Current Input Summary")
summary_df = pd.DataFrame({
    "Sensor": ["Air Temperature", "Rotational Speed", "Torque", "Tool Wear"],
    "Value": [f"{air_temp} K", f"{rot_speed} RPM", f"{torque} Nm", f"{tool_wear} min"],
    "Normal Range": ["295–305 K", "1168–2886 RPM", "3.8–76.6 Nm", "0–253 min"],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.75rem; margin-top:2rem;'>
    Model: Random Forest (n=100) · SMOTE Rebalancing · AUC-ROC: 0.95 · Recall: 76% · AI4I 2020 Dataset
</div>
""", unsafe_allow_html=True)

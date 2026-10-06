import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="🌸 Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ปรับแต่งโทนสีสดใสด้วย Custom CSS
st.markdown("""
<style>
    /* พื้นหลังหลักและฟอนต์ */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Sarabun', sans-serif;
    }
    
    /* สไตล์ Sidebar ให้มีสีสันสดใส */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #d4fc79 0%, #96e6a1 100%);
        box-shadow: 2px 0px 10px rgba(0,0,0,0.05);
    }
    
    /* กล่องผลลัพธ์การทำนาย (Prediction Result Card) */
    .result-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
        margin-bottom: 25px;
    }
    .result-card h3 {
        font-size: 1.1rem;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    .result-card h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        color: #ffffff;
    }
    .result-card p {
        margin-top: 10px;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* ปุ่มกด */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(221, 36, 118, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# 3. เตรียมข้อมูลและสร้างโมเดล Machine Learning
@st.cache_resource
def get_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    feature_means = X.mean(axis=0)
    return clf, iris.target_names, feature_means

model, class_names, avg_features = get_model()

# 4. ส่วน Sidebar: Input Sliders
st.sidebar.markdown("## 📊 Input Features")
st.sidebar.caption("Adjust the sliders to input flower measurements:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  min_value=2.0, max_value=4.5, value=3.4, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  min_value=0.1, max_value=2.5, value=1.2, step=0.1)

# ปุ่มทำนาย
predict_clicked = st.sidebar.button("Predict Species", use_container_width=True)

# ข้อมูลจาก Slider
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# คำนวณผลลัพธ์
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
pred_species = class_names[prediction].capitalize()
confidence = probabilities[prediction] * 100

# 5. ส่วนแสดงผลหลัก (Main Content)
st.markdown("# 🌸 Iris Flower Classifier")
st.markdown("##### Predict the species of Iris flowers using Machine Learning")
st.divider()

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("📈 Input Visualization")
    st.caption("Your Input vs Dataset Average")
    
    # กราฟแท่งเปรียบเทียบค่าเฉลี่ย
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    user_vals = [sepal_length, sepal_width, petal_length, petal_width]
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Your Input',
        x=features,
        y=user_vals,
        marker_color='#00b4d8', # ฟ้าสดใส
        text=user_vals,
        textposition='auto'
    ))
    fig_comp.add_trace(go.Bar(
        name='Dataset Average',
        x=features,
        y=np.round(avg_features, 2),
        marker_color='#7209b7', # ม่วงสดใส
        text=np.round(avg_features, 2),
        textposition='auto'
    ))
    
    fig_comp.update_layout(
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    
    # การ์ดแสดงผลสรุป
    st.markdown(f"""
    <div class="result-card">
        <h3>Predicted Species</h3>
        <h1>{pred_species}</h1>
        <p>Confidence: <b>{confidence:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Probability Distribution")
    
    # กราฟความน่าจะเป็น (Probability Bar Chart)
    species_labels = [name.capitalize() for name in class_names]
    prob_vals = [round(p * 100, 1) for p in probabilities]
    
    # กำหนดสีแท่งไฮไลต์เฉพาะตัวที่ชนะ
    colors = ['#FF6B6B' if s == pred_species else '#a0c4ff' for s in species_labels]
    
    fig_prob = go.Figure(go.Bar(
        x=species_labels,
        y=prob_vals,
        marker_color=colors,
        text=[f"{p}%" if p > 0 else "" for p in prob_vals],
        textposition='outside'
    ))
    fig_prob.update_layout(
        yaxis_title="Probability (%)",
        yaxis_range=[0, 105],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_prob, use_container_width=True)
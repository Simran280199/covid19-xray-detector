import streamlit as st
import numpy as np
import pathlib, os, json, io, urllib.request
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="COVID-19 X-Ray AI Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.hero{background:linear-gradient(135deg,#1d3557 0%,#457b9d 50%,#1d3557 100%);
      padding:2.5rem 2rem;border-radius:16px;margin-bottom:1.5rem;
      text-align:center;box-shadow:0 8px 32px rgba(69,123,157,0.3)}
.hero h1{color:#fff;font-size:2.2rem;font-weight:700;margin:0}
.hero p{color:#a8d8ea;font-size:1rem;margin:0.5rem 0 0}
.result-box{border-radius:12px;padding:1.5rem;margin:1rem 0;
            display:flex;align-items:center;gap:1rem}
.result-covid {background:rgba(231,76,60,0.15); border:2px solid #e74c3c}
.result-normal{background:rgba(46,204,113,0.15);border:2px solid #2ecc71}
.result-viral {background:rgba(230,126,34,0.15); border:2px solid #e67e22}
.result-icon{font-size:3rem}
.result-label{font-size:1.6rem;font-weight:700}
.result-conf{font-size:0.95rem;opacity:0.8;margin-top:0.2rem}
.section-header{font-size:1.2rem;font-weight:600;color:#cdd6f4;
                padding:0.5rem 0;border-bottom:2px solid #4361ee;margin-bottom:1rem}
.warn-box{background:rgba(231,76,60,0.1);border-left:4px solid #e74c3c;
          padding:1rem;border-radius:0 8px 8px 0;margin:0.5rem 0}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES  = ['Covid', 'Normal', 'Viral Pneumonia']
IMG_SIZE = 128
COLORS   = {'Covid':'#e74c3c','Normal':'#2ecc71','Viral Pneumonia':'#e67e22'}
ICONS    = {'Covid':'🦠','Normal':'✅','Viral Pneumonia':'⚠️'}

DATASET_STATS = {
    'Covid':           {'train':111,'test':26},
    'Normal':          {'train':70, 'test':20},
    'Viral Pneumonia': {'train':70, 'test':20},
}
TOTAL_TRAIN = sum(v['train'] for v in DATASET_STATS.values())
TOTAL_TEST  = sum(v['test']  for v in DATASET_STATS.values())

ALL_MODEL_RESULTS = pd.DataFrame([
    {'Model':'CNN Basic', 'Test Acc %':84.85,'F1 Score':0.8469,'Overfitting':'Yes',    'Params':'~1.2M'},
    {'Model':'ResNet50',  'Test Acc %':80.30,'F1 Score':0.8080,'Overfitting':'Low',    'Params':'~25.6M'},
    {'Model':'VGG16',     'Test Acc %':68.18,'F1 Score':0.6435,'Overfitting':'Low',    'Params':'~14.7M'},
    {'Model':'Deep CNN',  'Test Acc %':66.67,'F1 Score':0.6029,'Overfitting':'Reduced','Params':'~2.1M'},
])

# ── Model loading — tries local first, then Kaggle ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(model_file):
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    local_path = pathlib.Path(model_file)
    if local_path.exists():
        return tf.keras.models.load_model(str(local_path)), "local"

    # Try to download from Kaggle using secrets
    try:
        kaggle_user = st.secrets.get("KAGGLE_USERNAME", "")
        kaggle_key  = st.secrets.get("KAGGLE_KEY", "")
        if kaggle_user and kaggle_key:
            import kaggle
            kdir = pathlib.Path.home() / '.kaggle'
            kdir.mkdir(parents=True, exist_ok=True)
            (kdir/'kaggle.json').write_text(
                json.dumps({"username": kaggle_user, "key": kaggle_key}))
            (kdir/'kaggle.json').chmod(0o600)
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'pranavraikokte/covid19-image-dataset',
                path='.', unzip=True, quiet=True
            )
            if local_path.exists():
                return tf.keras.models.load_model(str(local_path)), "kaggle"
    except Exception as e:
        pass

    return None, "not_found"

def preprocess(img, model_key):
    arr = np.array(img.convert('RGB').resize((IMG_SIZE,IMG_SIZE)),
                   dtype='float32')
    if 'ResNet50' in model_key:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return np.expand_dims(preprocess_input(arr), 0)
    return np.expand_dims(arr/255.0, 0)

# ── Charts ────────────────────────────────────────────────────────────────────
def confidence_chart(probs):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    colors = [COLORS[c] for c in CLASSES]
    bars   = ax.barh(CLASSES, probs*100, color=colors, height=0.45)
    for bar,col in zip(bars,colors):
        ax.barh(bar.get_y()+bar.get_height()/2, bar.get_width(),
                height=0.6, color=col, alpha=0.15)
    ax.set_xlim(0, 110)
    for bar,v in zip(bars, probs*100):
        ax.text(v+1.5, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=11,
                fontweight='700', color='white')
    ax.set_xlabel('Confidence (%)', color='#8892b0', fontsize=9)
    ax.tick_params(colors='#cdd6f4', labelsize=10)
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig

def dataset_chart():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor('#1e2130')
    cls_names = list(DATASET_STATS.keys())
    colors    = [COLORS[c] for c in cls_names]
    for ax, split, title in zip(axes, ['train','test'], ['Training Set','Test Set']):
        ax.set_facecolor('#1e2130')
        vals = [DATASET_STATS[c][split] for c in cls_names]
        bars = ax.bar(cls_names, vals, color=colors, width=0.5)
        ax.set_title(title, color='#cdd6f4', fontsize=11, fontweight='600')
        ax.set_ylabel('Images', color='#8892b0', fontsize=9)
        ax.tick_params(colors='#cdd6f4', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2d3250')
        ax.spines['bottom'].set_color('#2d3250')
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, str(v),
                    ha='center', fontsize=10, fontweight='700', color='white')
    plt.tight_layout()
    return fig

def pie_chart():
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#1e2130')
    sizes  = [DATASET_STATS[c]['train'] for c in CLASSES]
    colors = [COLORS[c] for c in CLASSES]
    wedges,texts,autotexts = ax.pie(
        sizes, labels=CLASSES, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='#1e2130', linewidth=2))
    for t in texts:     t.set_color('#cdd6f4'); t.set_fontsize(9)
    for t in autotexts: t.set_color('white');  t.set_fontsize(9); t.set_fontweight('700')
    ax.set_title('Training Class Distribution', color='#cdd6f4',
                 fontsize=11, fontweight='600')
    plt.tight_layout()
    return fig

def model_chart():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    models   = ALL_MODEL_RESULTS['Model'].tolist()
    accs     = ALL_MODEL_RESULTS['Test Acc %'].tolist()
    bar_cols = ['#4fc3f7','#4361ee','#e67e22','#9b59b6']
    bars = ax.bar(models, accs, color=bar_cols, width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Test Accuracy (%)', color='#8892b0', fontsize=9)
    ax.set_title('Model Performance Comparison', color='#cdd6f4',
                 fontsize=11, fontweight='600')
    ax.tick_params(colors='#cdd6f4', labelsize=9)
    ax.axhline(y=80, color='white', lw=0.5, ls='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2d3250'); ax.spines['bottom'].set_color('#2d3250')
    for b,v in zip(bars,accs):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
                ha='center', fontsize=10, fontweight='700', color='white')
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫁 COVID-19 AI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Predict",
        "📊  EDA Dashboard",
        "🤖  Model Analysis",
        "📁  Batch Predict",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Select Model")
    model_choice = st.selectbox("", [
        "CNN Basic  ▸  84.85%",
        "ResNet50   ▸  80.30%",
    ], label_visibility="collapsed")

    model_file = ('covid_cnn_model.h5'
                  if 'CNN' in model_choice
                  else 'best_resnet_model.h5')

    st.markdown("---")
    st.markdown("### 📋 Dataset")
    st.markdown(f"""
    - **Classes:** 3
    - **Train:** {TOTAL_TRAIN} images
    - **Test:**  {TOTAL_TEST} images
    - **Size:** {IMG_SIZE}×{IMG_SIZE}px
    """)
    st.markdown("---")
    st.caption("⚠️ Research use only")

# ── Load model ────────────────────────────────────────────────────────────────
model, source = load_model_cached(model_file)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if '🏠' in page:
    st.markdown("""
    <div class="hero">
        <h1>🫁 COVID-19 X-Ray AI Detector</h1>
        <p>Deep Learning — CNN & Transfer Learning | 84.85% Accuracy</p>
    </div>""", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Best Accuracy", "84.85%")
    m2.metric("Train Images",  TOTAL_TRAIN)
    m3.metric("Models Trained","4")
    m4.metric("Classes",       "3")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown('<p class="section-header">📤 Upload Chest X-Ray</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=['jpg','jpeg','png'],
                                    label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, use_column_width=True, caption="Uploaded X-Ray")
            c1,c2,c3 = st.columns(3)
            c1.metric("Width",  f"{img.width}px")
            c2.metric("Height", f"{img.height}px")
            c3.metric("Size",   f"{uploaded.size//1024} KB")
        else:
            st.markdown("""
            <div style="border:2px dashed #2d3250;border-radius:12px;padding:3rem;
                        text-align:center;color:#8892b0;">
                <div style="font-size:3rem">🫁</div>
                <div style="margin-top:0.5rem">Drop X-Ray image here</div>
                <div style="font-size:0.8rem;margin-top:0.3rem">JPG, JPEG, PNG</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">🔍 AI Diagnosis</p>',
                    unsafe_allow_html=True)
        if model is None:
            st.error("❌ Model not found. Upload `covid_cnn_model.h5` to the repo.")
            st.info("For Streamlit Cloud: Add KAGGLE_USERNAME and KAGGLE_KEY to Secrets.")
        elif uploaded:
            with st.spinner("🧠 Analysing..."):
                inp   = preprocess(img, model_choice)
                probs = model.predict(inp, verbose=0)[0]
                pred  = CLASSES[np.argmax(probs)]
                conf  = float(np.max(probs))*100

            css_map = {'Covid':'result-covid','Normal':'result-normal',
                       'Viral Pneumonia':'result-viral'}
            col_map = {'Covid':'#e74c3c','Normal':'#2ecc71','Viral Pneumonia':'#e67e22'}
            st.markdown(f"""
            <div class="result-box {css_map[pred]}">
                <div class="result-icon">{ICONS[pred]}</div>
                <div>
                    <div class="result-label" style="color:{col_map[pred]}">{pred}</div>
                    <div class="result-conf">Confidence: {conf:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

            fig = confidence_chart(probs)
            st.pyplot(fig, use_container_width=True); plt.close()

            st.markdown("**Probability Breakdown:**")
            for cls, p in zip(CLASSES, probs):
                ca,cb = st.columns([4,1])
                ca.progress(float(p), text=f"{ICONS[cls]} **{cls}**")
                cb.markdown(f"<div style='text-align:right;font-weight:700;"
                            f"padding-top:8px'>{p*100:.1f}%</div>",
                            unsafe_allow_html=True)
            st.markdown("")
            if pred == 'Covid' and conf > 70:
                st.markdown("""<div class="warn-box">
                🚨 <b>High COVID-19 probability.</b> Consult a doctor immediately.
                </div>""", unsafe_allow_html=True)
            elif pred == 'Normal':
                st.success("✅ X-Ray appears normal.")
            else:
                st.warning("⚠️ Viral Pneumonia detected. Consult a doctor.")
        else:
            st.info("👆 Upload an X-Ray to get prediction")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif '📊' in page:
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>',
                unsafe_allow_html=True)

    st.markdown("### 📋 Dataset Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Images", TOTAL_TRAIN+TOTAL_TEST)
    c2.metric("Train Set",    TOTAL_TRAIN)
    c3.metric("Test Set",     TOTAL_TEST)
    c4.metric("Classes",      3)
    c5.metric("Image Size",   f"{IMG_SIZE}×{IMG_SIZE}")

    st.markdown("---")
    st.markdown("### 📁 Class-wise Distribution")
    df_dist = pd.DataFrame([{
        'Class': cls,
        'Train': DATASET_STATS[cls]['train'],
        'Test':  DATASET_STATS[cls]['test'],
        'Total': DATASET_STATS[cls]['train']+DATASET_STATS[cls]['test'],
        'Train %': f"{DATASET_STATS[cls]['train']/TOTAL_TRAIN*100:.1f}%",
        'Test %':  f"{DATASET_STATS[cls]['test']/TOTAL_TEST*100:.1f}%",
    } for cls in CLASSES])
    st.dataframe(df_dist, use_container_width=True, hide_index=True)

    st.markdown("---")
    cb, cp = st.columns([3,2])
    with cb:
        st.markdown("### 📊 Train vs Test Distribution")
        st.pyplot(dataset_chart(), use_container_width=True); plt.close()
    with cp:
        st.markdown("### 🥧 Training Composition")
        st.pyplot(pie_chart(), use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("### ⚖️ Class Imbalance")
    st.markdown("""
    | Observation | Detail |
    |---|---|
    | **Imbalanced** | Covid (111) > Normal (70) = Viral Pneumonia (70) |
    | **Ratio** | Covid is 1.58× more than other classes |
    | **Impact** | Model may be biased towards Covid class |
    | **Mitigation** | Data augmentation during training |
    """)

    st.markdown("---")
    st.markdown("### 🖼️ Image Properties")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Input Size",   "128×128 px")
    c2.metric("Channels",     "RGB (3)")
    c3.metric("Formats",      "JPG/JPEG/PNG")
    c4.metric("Normalisation","÷255 → [0,1]")

    st.markdown("---")
    st.markdown("### 📖 Dataset Info")
    st.markdown("""
    - **Name:** Covid19-image-dataset
    - **Source:** [Kaggle — pranavraikokte](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
    - **License:** CC-BY-SA-4.0
    - **Classes:** Covid-19, Normal, Viral Pneumonia
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif '🤖' in page:
    st.markdown('<p class="section-header">🤖 Model Analysis</p>',
                unsafe_allow_html=True)

    st.markdown("### 📊 Performance Comparison")
    st.pyplot(model_chart(), use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("### 📋 Detailed Results")
    st.dataframe(ALL_MODEL_RESULTS, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🃏 Architecture Details")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #4fc3f7;
                    border-radius:12px;padding:1.2rem">
        <h4 style="color:#4fc3f7;margin:0">🏆 CNN Basic — 84.85%</h4>
        <hr style="border-color:#2d3250">
        <b>Params:</b> ~1.2M | <b>Overfitting:</b> Yes<br>
        <b>Epochs:</b> 15 | <b>Optimizer:</b> Adam<br><br>
        Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
        → Flatten → Dense(128) → Dropout(0.5) → Softmax(3)
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #4361ee;
                    border-radius:12px;padding:1.2rem">
        <h4 style="color:#4361ee;margin:0">🔵 ResNet50 — 80.30%</h4>
        <hr style="border-color:#2d3250">
        <b>Params:</b> ~25.6M | <b>Overfitting:</b> Low<br>
        <b>Epochs:</b> 15 | <b>LR:</b> 0.0001<br><br>
        ResNet50(ImageNet, frozen) → Flatten
        → Dense(128) → Dropout(0.5) → Softmax(3)
        </div>""", unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #e67e22;
                    border-radius:12px;padding:1.2rem;margin-top:1rem">
        <h4 style="color:#e67e22;margin:0">🟠 VGG16 — 68.18%</h4>
        <hr style="border-color:#2d3250">
        <b>Params:</b> ~14.7M | <b>Overfitting:</b> Low<br>
        <b>Epochs:</b> 15+10 (fine-tune last 4 layers)<br><br>
        VGG16(ImageNet) → Flatten
        → Dense(128) → Dropout(0.5) → Softmax(3)
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #9b59b6;
                    border-radius:12px;padding:1.2rem;margin-top:1rem">
        <h4 style="color:#9b59b6;margin:0">🟣 Deep CNN — 66.67%</h4>
        <hr style="border-color:#2d3250">
        <b>Params:</b> ~2.1M | <b>Overfitting:</b> Reduced<br>
        <b>Epochs:</b> 20 + EarlyStopping<br><br>
        Conv2D(32→64→128) → MaxPool × 3
        → Flatten → Dense(128) → Dropout(0.5) → Softmax(3)
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Training Config")
    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Image Size | 128×128 px |
    | Batch Size | 32 |
    | Loss | Categorical Cross-Entropy |
    | Val Split | 20% |
    | Early Stopping | patience=3 |
    | Augmentation | rotation, zoom, flip, shift |
    """)

    st.markdown("---")
    st.markdown("### 📌 Key Findings")
    st.markdown("""
    - **CNN Basic achieved highest accuracy (84.85%)** despite being simplest model
    - **ResNet50 best F1 (0.8080)** — best balance of precision and recall
    - **Data augmentation** reduced overfitting in Deep CNN significantly
    - **Class imbalance** (Covid > others) may have contributed to CNN Basic's performance
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif '📁' in page:
    st.markdown('<p class="section-header">📁 Batch X-Ray Analysis</p>',
                unsafe_allow_html=True)
    if model is None:
        st.error("❌ Model not found.")
        st.stop()

    batch = st.file_uploader("Upload multiple X-Rays",
                             type=['jpg','jpeg','png'],
                             accept_multiple_files=True,
                             label_visibility="collapsed")
    if batch:
        results = []
        bar  = st.progress(0, text="Analysing...")
        cols = st.columns(4)
        for i, f in enumerate(batch):
            im = Image.open(f).convert('RGB')
            pr = model.predict(preprocess(im, model_choice), verbose=0)[0]
            pd_= CLASSES[np.argmax(pr)]
            cf = float(np.max(pr))*100
            with cols[i%4]:
                st.image(im, caption=f"{ICONS[pd_]} {pd_} ({cf:.0f}%)",
                         use_column_width=True)
            results.append({
                'File':f.name,'Prediction':pd_,
                'Confidence %':round(cf,1),
                'Covid %':round(pr[0]*100,1),
                'Normal %':round(pr[1]*100,1),
                'Viral %':round(pr[2]*100,1),
                'Risk':('🔴 HIGH' if pd_=='Covid' and cf>70
                        else '🟡 MED' if pd_=='Covid' else '🟢 LOW'),
            })
            bar.progress((i+1)/len(batch))

        st.markdown("---")
        df = pd.DataFrame(results)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total",    len(df))
        c2.metric("Covid",    len(df[df['Prediction']=='Covid']))
        c3.metric("Normal",   len(df[df['Prediction']=='Normal']))
        c4.metric("Viral",    len(df[df['Prediction']=='Viral Pneumonia']))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                           "predictions.csv", "text/csv",
                           use_container_width=True)
    else:
        st.info("👆 Upload multiple X-Ray images for bulk analysis")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem;'>"
    "🫁 COVID-19 X-Ray AI | TensorFlow + Streamlit | "
    "CNN Basic 84.85% | ResNet50 80.30% | ⚠️ Research use only</p>",
    unsafe_allow_html=True)

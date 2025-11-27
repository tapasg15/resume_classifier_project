# app.py (Enhanced: Evaluation + SHAP explainability + Batch classification)
import streamlit as st
import numpy as np
import pandas as pd
import joblib, pickle, json, io, re
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from logger_config import logger

ROOT = Path(__file__).parent

st.set_page_config(page_title="Resume Classifier — Advanced", layout="wide", page_icon="📄")

# ---------------------- Utilities ----------------------
def try_load_pipeline():
    p = ROOT / "pipeline.pkl"
    if p.exists():
        try:
            logger.info(f"Attempting to load pipeline from {p}")
            return joblib.load(p)
        except Exception as e:
            logger.exception(f"Failed to load pipeline from {p}: {e}")
            try:
                with open(p, "rb") as fh:
                    return pickle.load(fh)
            except Exception as e2:
                logger.exception(f"Retry with pickle failed: {e2}")
                return None
    logger.warning("pipeline.pkl not found in project root")
    return None


def load_metrics():
    m = ROOT / "metrics.json"
    if m.exists():
        return json.loads(m.read_text())
    return None

def extract_text_from_uploaded(uploaded):
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    raw = uploaded.read()
    try:
        if name.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        elif name.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def preprocess_text(t):
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r'\n+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

# ---------------------- Load assets ----------------------
pipeline = try_load_pipeline()
metrics = load_metrics()

# ---------------------- Sidebar navigation ----------------------
st.sidebar.title("Controls")
page = st.sidebar.radio("Go to", ["Classify Single", "Batch Classify", "Model Evaluation", "Explainability (SHAP)"])
st.sidebar.markdown("---")
if pipeline is None:
    st.sidebar.error("No pipeline found. Place pipeline.pkl in project root.")
else:
    st.sidebar.success("Pipeline loaded")

# ---------------------- Header ----------------------
st.markdown("<h1 style='margin-bottom:6px'>📄 Resume Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#9fc0cc;margin-bottom:16px'>Upload or paste resume text. Pipeline auto-loaded from pipeline.pkl</div>", unsafe_allow_html=True)

# ---------------------- PAGE: Classify Single ----------------------
if page == "Classify Single":
    col1, col2 = st.columns([3,1])
    with col1:
        st.header("1) Upload or paste resume")
        uploaded = st.file_uploader("Upload resume (PDF/TXT/DOCX)", type=["pdf","txt","docx"])
        st.write("or paste text below:")
        text_area = st.text_area("Paste resume text (optional)", height=260)
        exp = st.number_input("Years of experience", min_value=0, max_value=50, value=1)
        skills = st.text_input("Key skills (comma separated)", placeholder="python, sql, machine learning")
        if st.button("🔍 Classify (single)"):
            # get text
            if uploaded:
                text = extract_text_from_uploaded(uploaded)
            else:
                text = text_area
            text = preprocess_text(text)
            if not text:
                st.error("Please upload a resume or paste text.")
            else:
                with st.spinner("Predicting..."):
                    try:
                        if pipeline is not None:
                            pred = pipeline.predict([text])[0]
                            prob = None
                            if hasattr(pipeline, "predict_proba"):
                                prob_arr = pipeline.predict_proba([text])[0]
                                prob = float(np.max(prob_arr))
                                classes = list(pipeline.classes_) if hasattr(pipeline, "classes_") else None
                            st.success(f"Predicted: {pred}")
                            if prob is not None:
                                st.info(f"Confidence: {prob*100:.2f}%")
                        else:
                            # demo fallback
                            kw = ["data","machine learning","python","sql","react","aws"]
                            score = sum([1 for k in kw if k in text])
                            prob = 1 - np.exp(-score/5.0)
                            label = "Strong Match" if prob>0.6 else ("Good Match" if prob>0.35 else "Weak Match")
                            st.success(f"Demo label: {label}")
                            st.info(f"Score: {prob*100:.2f}%")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

    with col2:
        st.header("Preview & Quick info")
        if uploaded:
            st.write("Uploaded file:", uploaded.name)
            preview = extract_text_from_uploaded(uploaded)[:1500]
            st.code(preview if preview else "Preview not available", language='text')
        else:
            st.info("Upload a file to preview here or use the sample resumes.")
        st.markdown("---")
        st.subheader("Quick samples")
        if st.button("Load sample: Data Scientist"):
            st.session_state["sample_text"] = "Experienced data scientist with python, sklearn, pandas, machine learning experience."
            st.experimental_rerun()
        if "sample_text" in st.session_state:
            st.code(st.session_state["sample_text"])

# ---------------------- PAGE: Batch Classify ----------------------
elif page == "Batch Classify":
    st.header("Batch Classification")
    st.write("Upload multiple resume files (PDF/TXT/DOCX) OR upload a CSV with column 'text'.")
    multi = st.file_uploader("Upload multiple files or a CSV", accept_multiple_files=True, type=["pdf","txt","docx","csv"])
    if multi and st.button("Run batch classification"):
        rows = []
        for f in multi:
            fname = f.name
            if fname.lower().endswith(".csv"):
                # read CSV and expect 'text' column
                df_in = pd.read_csv(io.BytesIO(f.read()))
                if 'text' not in df_in.columns:
                    st.error(f"{fname} missing 'text' column")
                    continue
                for idx, row in df_in.iterrows():
                    txt = preprocess_text(str(row['text']))
                    if pipeline is not None:
                        try:
                            pred = pipeline.predict([txt])[0]
                            prob = None
                            if hasattr(pipeline, "predict_proba"):
                                prob = float(np.max(pipeline.predict_proba([txt])[0]))
                        except Exception:
                            pred, prob = "ERROR", None
                    else:
                        # demo
                        score = sum([1 for k in ["data","python","machine learning"] if k in txt])
                        prob = 1 - np.exp(-score/5.0)
                        pred = "Strong" if prob>0.6 else "Good" if prob>0.35 else "Weak"
                    rows.append({"filename": fname, "text_snippet": txt[:200], "pred": pred, "prob": prob})
            else:
                txt = preprocess_text(extract_text_from_uploaded(f))
                if pipeline is not None:
                    try:
                        pred = pipeline.predict([txt])[0]
                        prob = None
                        if hasattr(pipeline, "predict_proba"):
                            prob = float(np.max(pipeline.predict_proba([txt])[0]))
                    except Exception:
                        pred, prob = "ERROR", None
                else:
                    score = sum([1 for k in ["data","python","machine learning"] if k in txt])
                    prob = 1 - np.exp(-score/5.0)
                    pred = "Strong" if prob>0.6 else "Good" if prob>0.35 else "Weak"
                rows.append({"filename": f.name, "text_snippet": txt[:200], "pred": pred, "prob": prob})
        if rows:
            out_df = pd.DataFrame(rows)
            st.write("Results:")
            st.dataframe(out_df)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")

# ---------------------- PAGE: Model Evaluation ----------------------
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    if metrics is None:
        st.error("No metrics.json found. Run train_model.py to save metrics.json.")
    else:
        st.write("Accuracy:", metrics.get("accuracy"))
        # show classification report
        report = metrics.get("report")
        if report:
            st.subheader("Classification Report")
            # convert to dataframe for neat display
            rows = []
            for cls, vals in report.items():
                if isinstance(vals, dict):
                    r = {"class": cls}
                    r.update({k: vals.get(k) for k in ["precision","recall","f1-score","support"]})
                    rows.append(r)
            if rows:
                st.table(pd.DataFrame(rows).set_index("class"))
        # confusion matrix
        cm = metrics.get("confusion_matrix")
        classes = metrics.get("classes")
        if cm:
            cm_arr = np.array(cm)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, clear_figure=True)

# ---------------------- PAGE: Explainability (SHAP) ----------------------
elif page == "Explainability (SHAP)":
    st.header("Explainability (SHAP) - Top contributing words")
    st.write("This requires `shap` installed. If not installed, fallback to top TF-IDF words.")
    if pipeline is None:
        st.error("No pipeline found. Place pipeline.pkl in root.")
    else:
        text_for_explain = st.text_area("Paste resume text to explain", height=250)
        if st.button("Explain"):
            t = preprocess_text(text_for_explain)
            if not t:
                st.error("Paste some text to explain.")
            else:
                # Try SHAP
                try:
                    import shap
                    st.info("Running SHAP explainer (may take a few seconds)...")
                    # If pipeline is sklearn pipeline with named_steps
                    if hasattr(pipeline, "named_steps") and 'tfidfvectorizer' in pipeline.named_steps:
                        tf = pipeline.named_steps.get('tfidfvectorizer') or pipeline.named_steps.get('tfidf')
                        clf = None
                        # pick classifier step
                        for k,v in pipeline.named_steps.items():
                            if k != 'tfidfvectorizer' and k != 'tfidf':
                                clf = v
                                break
                        if clf is None:
                            st.warning("Could not detect classifier step in pipeline for SHAP.")
                        else:
                            # shap expects a function that maps raw input to model output; using kernel explainer for text is fine for demonstration
                            explainer = shap.Explainer(clf, tf.transform)
                            shap_vals = explainer([t])
                            # show bar plot
                            st.subheader("SHAP values (top features)")
                            shap.plots.bar(shap_vals[0], show=False)
                            st.pyplot(bbox_inches='tight')
                    else:
                        # fallback: use shap on pipeline by wrapping predict function
                        explainer = shap.Explainer(pipeline.predict, pipeline.transform if hasattr(pipeline, "transform") else None)
                        shap_vals = explainer([t])
                        st.subheader("SHAP values (top features)")
                        shap.plots.bar(shap_vals[0], show=False)
                        st.pyplot(bbox_inches='tight')
                except Exception as e:
                    st.warning("SHAP not available or failed. Showing top TF-IDF words instead.")
                    # fallback: try to show top tfidf words if pipeline has vectorizer
                    try:
                        vec = None
                        if hasattr(pipeline, "named_steps"):
                            for name, step in pipeline.named_steps.items():
                                if "tfidf" in name.lower():
                                    vec = step
                                    break
                        if vec is None:
                            st.info("No TF-IDF vectorizer found inside pipeline.")
                        else:
                            v = vec.transform([t])
                            arr = v.toarray()[0]
                            try:
                                names = vec.get_feature_names_out()
                            except Exception:
                                names = vec.get_feature_names()
                            top_idx = np.argsort(arr)[-20:][::-1]
                            top_words = [(names[i], float(arr[i])) for i in top_idx if arr[i] > 0]
                            st.write("Top TF-IDF words:")
                            st.table(pd.DataFrame(top_words, columns=["word","score"]))
                    except Exception as e2:
                        st.error("Explainability fallback failed: " + str(e2))

# app.py
# Fully-upgraded Resume Classifier app + lightweight Login/Signup
# (Everything from your original app preserved; added safe rerun helper and local auth)

import streamlit as st
import numpy as np
import pandas as pd
import joblib, pickle, json, io, re, os
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import datetime
import hashlib
import secrets

# optional/extra libraries (safe imports)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WordCloud = None
    WORDCLOUD_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# logger fallback
try:
    from logger_config import logger
except Exception:
    class _DummyLogger:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def exception(self, *a, **k): pass
    logger = _DummyLogger()

ROOT = Path(__file__).parent
HISTORY_FILE = ROOT / "history.json"
PIPELINE_FILE = ROOT / "pipeline.pkl"
METRICS_FILE = ROOT / "metrics.json"
USERS_FILE = ROOT / "users.json"    # <-- user store

st.set_page_config(page_title="Resume Classifier — Pro", layout="wide", page_icon="📄")

# ---------- small helper: safe rerun ----------
def safe_rerun():
    """
    Try to call st.experimental_rerun() if available.
    Otherwise mutate a dummy session_state key so Streamlit re-runs the script.
    """
    try:
        st.experimental_rerun()
    except Exception:
        # fallback: change a session_state value
        st.session_state["_dummy_rerun"] = st.session_state.get("_dummy_rerun", 0) + 1

# ---------- CSS & small animations ----------
st.markdown(
    """
    <style>
    :root {
        --bg:#0b0f12;
        --card:#0f1720;
        --muted:#93a3a8;
        --accent:#0b6e4f;
    }
    .app-header {
        font-size:28px; font-weight:800; margin-bottom:6px;
    }
    .muted { color:var(--muted); font-size:13px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:18px; border-radius:12px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); }
    .result-card { padding:14px; border-radius:10px; background: linear-gradient(90deg, #0b6e4f, #0e8b66); color:white; box-shadow: 0 6px 24px rgba(11,110,79,0.18); transform: translateY(0); transition: transform 0.18s ease; }
    .result-card:hover { transform: translateY(-6px); }
    .mono { font-family: monospace; background:#071018; padding:8px; border-radius:8px; }
    .small { font-size:13px; color:var(--muted); }
    .glow { box-shadow: 0 8px 30px rgba(11,110,79,0.12); }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='app-header'>📄 Resume Classifier — Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Paste resume text or upload a PDF/TXT/DOCX. Dashboard, export & explainability included.</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- Simple local auth helpers --------------------
# Password hashing using PBKDF2-HMAC-SHA256 (stdlib)
PBKDF2_ITERS = 100_000

def _hash_password(password: str, salt: bytes = None):
    """Return (salt_hex, hash_hex)"""
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return salt.hex(), key.hex()

def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return key.hex() == hash_hex

def load_users():
    try:
        if USERS_FILE.exists():
            return json.loads(USERS_FILE.read_text())
    except Exception as e:
        logger.exception("load_users failed: " + str(e))
    return {}

def save_users(users_dict):
    try:
        USERS_FILE.write_text(json.dumps(users_dict, indent=2))
    except Exception as e:
        logger.exception("save_users failed: " + str(e))

def create_user(username: str, password: str) -> (bool, str):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    salt_hex, hash_hex = _hash_password(password)
    users[username] = {"salt": salt_hex, "hash": hash_hex, "created": datetime.datetime.now().isoformat()}
    save_users(users)
    return True, "User created"

def authenticate_user(username: str, password: str) -> (bool, str):
    users = load_users()
    if username not in users:
        return False, "No such username"
    rec = users[username]
    if _verify_password(password, rec["salt"], rec["hash"]):
        return True, "Authenticated"
    return False, "Wrong password"

# Initialize session state for auth
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# -------------------- Sidebar: Login / Signup --------------------
with st.sidebar:
    st.header("Account")
    # If logged in -> show user info + logout
    if st.session_state["logged_in"]:
        st.markdown(f"**Signed in as:** `{st.session_state['username']}`")
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            safe_rerun()
    else:
        choice = st.selectbox("Choose:", ["Login", "Signup"])
        if choice == "Signup":
            new_user = st.text_input("Choose username")
            new_pass = st.text_input("Choose password", type="password")
            new_pass2 = st.text_input("Repeat password", type="password")
            if st.button("Create account"):
                if not new_user or not new_pass:
                    st.error("Provide username and password")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match")
                else:
                    ok, msg = create_user(new_user, new_pass)
                    if ok:
                        st.success("Account created — please login now")
                    else:
                        st.error(msg)
        else:
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.button("Login"):
                ok, msg = authenticate_user(user, pwd)
                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = user
                    st.success("Logged in as " + user)
                    safe_rerun()
                else:
                    st.error(msg)
    st.markdown("---")
    st.write("Tips & Tools")
    st.write("WordCloud:", "✅" if WORDCLOUD_AVAILABLE else "❌")
    st.write("SHAP:", "✅" if SHAP_AVAILABLE else "❌")
    st.write("PDF Export:", "✅" if REPORTLAB_AVAILABLE else "❌")
    st.markdown("---")
    st.write("Need to reset demo users? Delete `users.json` file.")

# --------------------- If not logged in, block pages (friendly) ---------------------
if not st.session_state["logged_in"]:
    st.warning("Please signup/login from the sidebar to use the app. (Local account only for demo.)")
    st.stop()  # stops execution here until user logs in

# =========================== (rest of the app remains same) ===========================
# ---------- Utilities (app internals) ----------
def try_load_pipeline():
    if PIPELINE_FILE.exists():
        try:
            logger.info(f"Loading pipeline from {PIPELINE_FILE}")
            return joblib.load(PIPELINE_FILE)
        except Exception as e:
            logger.exception("joblib load failed: " + str(e))
            try:
                with open(PIPELINE_FILE, "rb") as fh:
                    return pickle.load(fh)
            except Exception as e2:
                logger.exception("pickle load failed: " + str(e2))
                return None
    return None

def load_metrics():
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text())
        except:
            return None
    return None

def save_history(entry):
    try:
        data = []
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text())
        data.insert(0, entry)  # newest first
        # keep last 200
        data = data[:200]
        HISTORY_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.exception("Failed to save history: " + str(e))

def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except:
            return []
    return []

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
        except:
            return ""

def preprocess_text(t):
    if not t:
        return ""
    t = str(t)
    t = t.lower()
    t = re.sub(r'\n+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return t.strip()

# ------------------ NEW: preprocessing helpers that match training ------------------
# These helpers mirror the `build_processed_text` logic used during training so
# inference matches training preprocessing (SKILL_ and YEARS_ tokens are appended).
COMMON_SKILLS_TOKENS = [s for s in [
    "python","java","c++","c#","sql","mysql","postgresql","mongodb",
    "html","css","javascript","react","node","django","flask",
    "power bi","excel","tableau","pandas","numpy","tensorflow","keras",
    "nlp","git","github","aws","docker","redis","kubernetes","jenkins","selenium"
] ]

def clean_text_basic(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'\S+@\S+', ' ', t)
    t = re.sub(r'\+?\d[\d\s\-\(\)]{6,}', ' ', t)
    t = t.lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_years_from_text(t):
    m = re.search(r'(\d{1,2})\s*(?:years|yrs|y)\b', t, flags=re.I)
    if m:
        try:
            y = int(m.group(1))
            if y < 0:
                return 0
            if y > 40:
                return 40
            return y
        except:
            return 0
    return 0

def extract_skills_list(t):
    t_low = t.lower()
    found = []
    for s in COMMON_SKILLS_TOKENS:
        if s in t_low:
            found.append(s.replace(' ', '_'))
    return sorted(set(found))

def build_processed_text(raw):
    cleaned = clean_text_basic(raw)
    yrs = extract_years_from_text(raw)
    skills = extract_skills_list(raw)
    extra = ""
    if skills:
        extra += " " + " ".join([f"SKILL_{s}" for s in skills])
    if yrs:
        extra += f" YEARS_{yrs}"
    return (cleaned + " " + extra).strip()

# -----------------------------------------------------------------
# Keep your existing COMMON_SKILLS (for display) unchanged so UI remains same
COMMON_SKILLS = [
    "python","java","c++","c#","sql","mysql","postgresql","mongodb",
    "html","css","javascript","react","node","django","flask",
    "power bi","excel","tableau","pandas","numpy","machine learning",
    "deep learning","tensorflow","keras","nlp","git","github","aws","docker"
]

def extract_skills(text):
    text_lower = (text or "").lower()
    found = [s for s in COMMON_SKILLS if s in text_lower]
    return sorted(set(found))

# rest of helper functions unchanged

def highlight_skills_in_text(text, skills):
    if not text:
        return ""
    safe = text.replace("<","&lt;").replace(">","&gt;")
    for skill in sorted(skills, key=len, reverse=True):
        pattern = re.compile(r"(?i)\b" + re.escape(skill) + r"\b")
        safe = pattern.sub(f"<mark style='background:#b6f5c3;color:#000;padding:0 3px;border-radius:3px'>{skill}</mark>", safe)
    return safe

def show_wordcloud_from_skills(skill_list):
    if not WORDCLOUD_AVAILABLE:
        st.info("WordCloud not installed. Install `wordcloud` to enable this.")
        return
    text = " ".join(skill_list)
    if not text.strip():
        st.info("No skills to draw a word cloud.")
        return
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(9,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Resume scoring (explainable)
def resume_score(text, skills=None, years_of_exp=0):
    if not text:
        return 0, {"skills":0,"experience":0,"length":0}
    s = skills or extract_skills(text)
    skill_pts = min(50, len(s) * 10)
    exp_pts = min(30, int(years_of_exp) * 2)  # change weighting: up to 30
    length_pts = min(20, len(text.split()) // 50)
    score = skill_pts + exp_pts + length_pts
    return int(min(100, score)), {"skills": skill_pts, "experience": exp_pts, "length": length_pts}

# Suggestion engine (simple rules)
def suggestions_from_text(text, skills_list, predicted):
    sugs = []
    txt = text.lower()
    if len(skills_list) < 3:
        sugs.append("Add clear technical skills (e.g., Python, SQL, React) in a separate skills section.")
    if "github" not in txt and any(s in txt for s in ["python","javascript","react"]):
        sugs.append("Add your GitHub link or projects (helps for developer roles).")
    if "project" not in txt and any(s in ["machine learning","data","pandas"] for s in skills_list):
        sugs.append("Mention project titles and a brief achievement bullet for each project.")
    if "year" not in txt and "experience" in txt:
        sugs.append("Add years of experience (e.g., '2 years experience') for quick screening.")
    if predicted and predicted.lower() in ["web developer","software developer"] and "portfolio" not in txt:
        sugs.append("Include a portfolio/demo link for developer roles (optional).")
    if not sugs:
        sugs.append("Nice! Resume looks good. Consider adding metrics (numbers) to achievements.")
    return sugs

# Export report (PDF if possible else fallback to HTML)
def make_report_bytes(title, predicted, confidence, top_skills, score_breakdown, highlighted_html, charts_bytes=None):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"resume_report_{now}.pdf"
    if REPORTLAB_AVAILABLE:
        bio = BytesIO()
        doc = SimpleDocTemplate(bio, pagesize=letter)
        styles = getSampleStyleSheet()
        flow = []
        flow.append(Paragraph(f"<b>{title}</b>", styles['Title']))
        flow.append(Spacer(1,8))
        flow.append(Paragraph(f"Predicted: <b>{predicted}</b> (Confidence: {confidence:.2f}%)", styles['Normal']))
        flow.append(Spacer(1,6))
        flow.append(Paragraph("<b>Top skills:</b> " + ", ".join(top_skills), styles['Normal']))
        flow.append(Spacer(1,6))
        flow.append(Paragraph("<b>Score breakdown:</b>", styles['Normal']))
        for k, v in score_breakdown.items():
            flow.append(Paragraph(f"- {k}: {v}", styles['Normal']))
        flow.append(Spacer(1,10))
        flow.append(Paragraph("<b>Resume (highlights):</b>", styles['Normal']))
        flow.append(Spacer(1,6))
        text_plain = re.sub(r'<[^>]+>', '', highlighted_html)
        for chunk in (text_plain[i:i+900] for i in range(0, len(text_plain), 900)):
            flow.append(Paragraph(chunk.replace("\n", "<br/>"), styles['BodyText']))
            flow.append(Spacer(1,4))
        doc.build(flow)
        bio.seek(0)
        return filename, bio.read(), "application/pdf"
    else:
        html = f"""
        <html><head><meta charset="utf-8"><title>{title}</title></head><body>
        <h1>{title}</h1>
        <h3>Predicted: {predicted} (Confidence: {confidence:.2f}%)</h3>
        <h4>Top skills: {', '.join(top_skills)}</h4>
        <h4>Score breakdown:</h4><ul>
        """
        for k, v in score_breakdown.items():
            html += f"<li>{k}: {v}</li>"
        html += "</ul><hr/><h4>Resume (highlighted)</h4>"
        html += highlighted_html
        html += "</body></html>"
        return f"resume_report_{now}.html", html.encode("utf-8"), "text/html"

# ---------- Load pipeline / metrics ----------
pipeline = try_load_pipeline()
metrics = load_metrics()

# ---------- Sidebar: Quick Controls (after login) ----------
with st.sidebar:
    st.header(f"Welcome, {st.session_state['username']} ✅")
    st.write("Quick Controls")
    page = st.radio("Choose page:", ["Classify Single", "Batch Classify", "Dashboard", "Model Eval", "Explainability"])
    st.markdown("---")
    st.write("Pipeline loaded: ", "✅" if pipeline is not None else "❌ (place pipeline.pkl)")
    if st.button("Reload pipeline"):
        pipeline = try_load_pipeline()
        safe_rerun()

# ---------------- Classify Single ----------------
if page == "Classify Single":
    st.subheader("Classify a single resume")
    left, right = st.columns([3,1])
    with left:
        uploaded = st.file_uploader("Upload resume (PDF/TXT/DOCX)", type=["pdf","txt","docx"])
        text_area = st.text_area("Or paste resume text here (optional)", height=280)
        if "sample_text" in st.session_state and not text_area:
            text_area = st.session_state["sample_text"]
        experience = st.number_input("Years of experience (optional)", min_value=0, max_value=50, value=0)
        manual_skill_input = st.text_input("Manual skills (comma separated) - optional")
        if st.button("Classify"):
            raw_text = (extract_text_from_uploaded(uploaded) if uploaded else text_area) or ""
            # Use the same preprocessing as used in training
            processed_text = build_processed_text(raw_text)
            if not processed_text:
                st.error("Please provide resume text or upload a file.")
            else:
                with st.spinner("Classifying..."):
                    # predictive step
                    predicted = None
                    confidence = 0.0
                    classes = None
                    probs = None
                    try:
                        if pipeline is not None:
                            predicted = pipeline.predict([processed_text])[0]
                            if hasattr(pipeline, "predict_proba"):
                                probs = pipeline.predict_proba([processed_text])[0]
                                confidence = float(np.max(probs))*100
                            if hasattr(pipeline, "classes_"):
                                classes = list(pipeline.classes_)
                        else:
                            demo_score = sum([1 for k in ["python","sql","react","machine learning","aws"] if k in processed_text])
                            predicted = "Data Analyst" if "data" in processed_text else ("Web Developer" if "html" in processed_text or "css" in processed_text else "Software Developer")
                            confidence = min(90, 40 + demo_score*10)
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))
                        predicted = "ERROR"
                    # skills (display uses raw_text for readability)
                    if manual_skill_input:
                        skills_list = [s.strip() for s in manual_skill_input.split(",") if s.strip()]
                    else:
                        skills_list = extract_skills(raw_text)
                    # score
                    score_val, breakdown = resume_score(raw_text, skills_list, years_of_exp=experience)
                    # suggestions
                    sugs = suggestions_from_text(raw_text, skills_list, predicted)

                    # show results nicely
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.markdown(f"<div class='result-card glow'><b>Predicted:</b> {predicted}</div>", unsafe_allow_html=True)
                        st.write("")
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                        st.markdown("**Top skills detected:** " + (", ".join(skills_list) if skills_list else "—"))
                        st.markdown("### ATS-style Score")
                        st.progress(score_val)
                        st.write(f"**{score_val}/100**")
                        st.write("Breakdown:")
                        st.write(f"- Skills points: {breakdown['skills']}")
                        st.write(f"- Experience points: {breakdown['experience']}")
                        st.write(f"- Length points: {breakdown['length']}")

                        st.markdown("---")
                        st.markdown("### Suggestions to improve resume")
                        for s in sugs:
                            st.write("• " + s)

                        # Save history entry
                        try:
                            entry = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "predicted": predicted,
                                "confidence": float(confidence),
                                "top_skills": skills_list,
                                "score": int(score_val),
                                "user": st.session_state['username']
                            }
                            save_history(entry)
                        except Exception:
                            pass

                        # Export
                        highlighted_html = highlight_skills_in_text(raw_text, skills_list) if skills_list else raw_text
                        if st.button("Export report (PDF/HTML)"):
                            fname, bts, mime = make_report_bytes("Resume Analysis", predicted, confidence, skills_list, breakdown, highlighted_html)
                            st.success(f"Prepared {fname}")
                            st.download_button("Download Report", data=bts, file_name=fname, mime=mime)

                    with col2:
                        st.header("Resume preview")
                        if raw_text:
                            if st.checkbox("Show highlighted view", value=True):
                                st.markdown(highlighted_html, unsafe_allow_html=True)
                            else:
                                st.code(raw_text[:3000], language='text')
                        else:
                            st.info("No resume text available to preview.")

                        if st.checkbox("Show wordcloud of skills", value=False):
                            show_wordcloud_from_skills(skills_list)

    with right:
        st.header("Quick samples & tools")
        if st.button("Load sample: Data Analyst"):
            st.session_state["sample_text"] = "Data analyst with Python, Pandas, SQL, Power BI, Tableau, data visualization and storytelling."
            safe_rerun()
        if st.button("Load sample: Web Dev"):
            st.session_state["sample_text"] = "Web developer skilled in HTML, CSS, JavaScript, React, Node.js and Git."
            safe_rerun()
        st.markdown("---")
        st.write("Pipeline present:", "✅" if pipeline else "❌")
        st.write("WordCloud:", "✅" if WORDCLOUD_AVAILABLE else "❌")
        st.write("SHAP:", "✅" if SHAP_AVAILABLE else "❌")
        st.write("PDF Export (reportlab):", "✅" if REPORTLAB_AVAILABLE else "❌")

# ---------------- Batch Classify ----------------
elif page == "Batch Classify":
    st.subheader("Batch classify multiple resumes")
    uploaded = st.file_uploader("Upload multiple files or a single CSV with 'text' column", accept_multiple_files=True, type=["pdf","txt","docx","csv"])
    if uploaded and st.button("Run Batch"):
        rows = []
        progress = st.progress(0)
        total = len(uploaded)
        for i, f in enumerate(uploaded):
            try:
                if f.name.lower().endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(f.read()))
                    if "text" not in df.columns:
                        st.error("CSV must contain 'text' column")
                        continue
                    for _, r in df.iterrows():
                        raw_txt = str(r["text"])
                        txt = build_processed_text(raw_txt)
                        if pipeline:
                            try:
                                pred = pipeline.predict([txt])[0]
                                prob = float(np.max(pipeline.predict_proba([txt])[0])) if hasattr(pipeline, "predict_proba") else None
                            except:
                                pred, prob = "ERROR", None
                        else:
                            pred = "Data Analyst" if "data" in txt else "Web Developer" if "html" in txt else "Software Developer"
                            prob = None
                        skills = extract_skills(raw_txt)
                        score_val, _ = resume_score(raw_txt, skills)
                        rows.append({"filename": f.name, "pred": pred, "prob": prob, "skills": skills, "score": score_val})
                else:
                    txt_raw = extract_text_from_uploaded(f)
                    txt = build_processed_text(txt_raw)
                    if pipeline:
                        try:
                            pred = pipeline.predict([txt])[0]
                            prob = float(np.max(pipeline.predict_proba([txt])[0])) if hasattr(pipeline, "predict_proba") else None
                        except:
                            pred, prob = "ERROR", None
                    else:
                        pred = "Data Analyst" if "data" in txt else "Web Developer" if "html" in txt else "Software Developer"
                        prob = None
                    skills = extract_skills(txt_raw)
                    score_val, _ = resume_score(txt_raw, skills)
                    rows.append({"filename": f.name, "pred": pred, "prob": prob, "skills": skills, "score": score_val})
            except Exception as e:
                logger.exception("Batch item failed: " + str(e))
            progress.progress(int((i+1)/total * 100))

        if rows:
            df_out = pd.DataFrame(rows)
            st.write(df_out)
            st.download_button("Download CSV", df_out.to_csv(index=False).encode("utf-8"), "batch_results.csv", "text/csv")
            # analytics
            st.markdown("---")
            st.subheader("Batch Analytics")
            try:
                vc = df_out['pred'].value_counts()
                fig, ax = plt.subplots(figsize=(6,3))
                sns.barplot(x=vc.index, y=vc.values, ax=ax)
                ax.set_title("Category distribution")
                st.pyplot(fig)
            except Exception:
                st.info("Not enough data for charts.")

            # skills aggregation
            try:
                all_skills = [sk for sub in df_out['skills'] for sk in (sub if isinstance(sub, list) else [])]
                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(20)
                    st.subheader("Top skills in batch")
                    st.bar_chart(skill_counts)
                    if st.checkbox("Show wordcloud of aggregated skills", value=True):
                        show_wordcloud_from_skills(all_skills)
                else:
                    st.info("No skills found to aggregate.")
            except Exception as e:
                st.warning("Failed to aggregate skills: " + str(e))

# ---------------- Dashboard ----------------
elif page == "Dashboard":
    st.subheader("Dashboard")
    history = load_history()
    st.markdown("### Recent classifications")
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist.head(50))
        # summary
        st.markdown("---")
        st.subheader("Summary")
        try:
            categories = df_hist['predicted'].value_counts()
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x=categories.index, y=categories.values, ax=ax)
            ax.set_title("Predicted category counts")
            st.pyplot(fig)
        except Exception:
            pass
        # top skills overall
        try:
            all_sk = [s for sub in df_hist['top_skills'] for s in (sub if isinstance(sub, list) else [])]
            if all_sk:
                sc = pd.Series(all_sk).value_counts().head(20)
                st.subheader("Top skills (recent)")
                st.bar_chart(sc)
        except Exception:
            pass
        st.markdown("---")
        st.write("Total processed (history):", len(history))
    else:
        st.info("No history yet. Process resumes from the Classify pages to populate dashboard.")

# ---------------- Model Evaluation ----------------
elif page == "Model Eval":
    st.subheader("Model evaluation")
    if metrics is None:
        st.warning("No metrics.json found. Run training script to produce metrics.json")
    else:
        st.write("Accuracy:", metrics.get("accuracy"))
        report = metrics.get("report")
        if report:
            rows = []
            for cls, vals in report.items():
                if isinstance(vals, dict):
                    r = {"class": cls}
                    r.update({k: vals.get(k) for k in ["precision","recall","f1-score","support"]})
                    rows.append(r)
            if rows:
                st.table(pd.DataFrame(rows).set_index("class"))
        cm = metrics.get("confusion_matrix")
        classes = metrics.get("classes")
        if cm and classes:
            cm_arr = np.array(cm)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            st.pyplot(fig)

# ---------------- Explainability ----------------
elif page == "Explainability":
    st.subheader("Explainability")
    text = st.text_area("Paste resume text for explainability", height=240)
    if st.button("Explain"):
        # use same processed text for explainability to keep features aligned
        t = build_processed_text(text)
        if not t:
            st.error("Please paste some text")
        else:
            st.info("Attempting SHAP explainability (may be slow). If SHAP not installed, fallback to TF-IDF words.")
            try:
                if SHAP_AVAILABLE and pipeline is not None:
                    vec = None
                    clf = None
                    if hasattr(pipeline, "named_steps"):
                        for name, step in pipeline.named_steps.items():
                            if "tfidf" in name.lower():
                                vec = step
                            elif clf is None and "tfidf" not in name.lower():
                                clf = step
                    if clf is None:
                        explainer = shap.Explainer(pipeline.predict, pipeline.transform if hasattr(pipeline, "transform") else None)
                        vals = explainer([t])
                        st.write(vals)
                        shap.plots.bar(vals[0], show=False)
                        st.pyplot()
                    else:
                        explainer = shap.Explainer(clf, vec.transform if vec is not None else None)
                        vals = explainer([t])
                        shap.plots.bar(vals[0], show=False)
                        st.pyplot()
                else:
                    raise Exception("SHAP not available")
            except Exception as e:
                st.warning("SHAP not available or failed. Showing top TF-IDF words if possible.")
                vec = None
                if pipeline is not None and hasattr(pipeline, "named_steps"):
                    for name, step in pipeline.named_steps.items():
                        if "tfidf" in name.lower():
                            vec = step
                            break
                if vec is None:
                    st.info("No TF-IDF vectorizer found inside pipeline.")
                else:
                    try:
                        v = vec.transform([t])
                        arr = v.toarray()[0]
                        try:
                            names = vec.get_feature_names_out()
                        except:
                            names = vec.get_feature_names()
                        top_idx = np.argsort(arr)[-20:][::-1]
                        top_words = [(names[i], float(arr[i])) for i in top_idx if arr[i] > 0]
                        st.table(pd.DataFrame(top_words, columns=["word","tfidf"]))
                    except Exception as e2:
                        st.error("Explainability fallback failed: " + str(e2))

# train_model.py (robust beginner-friendly) — UPDATED with light preprocessing & feature tokens
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib, json, sys, re

ROOT = Path(".")
DATA_PATH = ROOT / "data" / "resumes.csv"   # change if your CSV is elsewhere

# ------------------- small feature lists -------------------
COMMON_SKILLS = [
    "python","java","c++","c#","sql","mysql","postgresql","mongodb",
    "html","css","javascript","react","node","django","flask",
    "power bi","excel","tableau","pandas","numpy","tensorflow","keras",
    "nlp","git","github","aws","docker","redis","kubernetes","jenkins","selenium"
]

# -------------------------------------------------------------------------
# 1) Load CSV safely and show columns if label column missing
# -------------------------------------------------------------------------
if not DATA_PATH.exists():
    print(f"ERROR: Dataset not found at {DATA_PATH}. Please put your CSV at this path.")
    sys.exit(1)

# Try common encodings if default fails
df = None
for enc in ["utf-8", "latin1", "utf-8-sig"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        break
    except Exception as e:
        df = None

if df is None:
    print("Failed to read CSV with common encodings. Open the file and check encoding.")
    sys.exit(1)

print("Loaded dataset. Columns:", df.columns.tolist())

# -------------------------------------------------------------------------
# 2) Detect label (target) column automatically
# -------------------------------------------------------------------------
possible_label_names = ["label", "target", "category", "class", "y", "tag"]
label_col = None
for name in possible_label_names:
    if name in df.columns:
        label_col = name
        break

# If not found, try heuristic: if there are exactly 2 columns, assume second is label
if label_col is None:
    if len(df.columns) == 2:
        label_col = df.columns[1]
        print(f"No common label name found — assuming second column '{label_col}' is the label.")
    else:
        print("\nCould not auto-detect label column.")
        print("Available columns:", df.columns.tolist())
        print("\nHere are the first 5 rows to help you choose:")
        print(df.head().to_string())
        print("\nSOLUTION OPTIONS:")
        print("1) Rename your label column to one of: ", possible_label_names)
        print("   e.g. open CSV and change header to 'label'")
        print("2) Or edit train_model.py and set label_col = '<your_column_name>' manually near the top.")
        sys.exit(1)

print("Using label column:", label_col)

# -------------------------------------------------------------------------
# 3) Prepare X and y (with light preprocessing + appended feature-tokens)
# -------------------------------------------------------------------------

# choose text column (same logic you had)
if "text" in df.columns:
    text_col = "text"
else:
    for cand in ["resume", "resume_text", "content", "description", "cv", "document"]:
        if cand in df.columns:
            text_col = cand
            break
    else:
        text_candidates = [c for c in df.columns if c != label_col]
        if len(text_candidates) >= 1:
            text_col = text_candidates[0]
            print(f"No explicit text column found — using '{text_col}' as text.")
        else:
            print("No text column found. Please ensure your CSV has a column with resume text.")
            sys.exit(1)

print("Using text column:", text_col)

df[text_col] = df[text_col].fillna("").astype(str)
df[label_col] = df[label_col].astype(str)

# ---------- helper preprocessing functions ----------
def clean_text_basic(t):
    if not isinstance(t, str):
        return ""
    # remove urls, emails, phones
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'\S+@\S+', ' ', t)
    t = re.sub(r'\+?\d[\d\s\-\(\)]{6,}', ' ', t)
    t = t.lower()
    # keep alnum and spaces
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_years(t):
    # simple regex: captures "3 years", "2 yrs", "5 Yrs" etc.
    m = re.search(r'(\d{1,2})\s*(?:years|yrs|y)\b', t, flags=re.I)
    if m:
        try:
            y = int(m.group(1))
            # cap to reasonable range
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
    for s in COMMON_SKILLS:
        # match whole words where possible
        if s.lower() in t_low:
            found.append(s.replace(" ", "_"))
    return sorted(set(found))

# Build processed text: cleaned text + appended tokens for skills and years
def build_processed_text(raw):
    cleaned = clean_text_basic(raw)
    yrs = extract_years(raw)
    skills = extract_skills_list(raw)
    # append skill tokens and a YEARS_x token so TF-IDF sees them as features
    extra = ""
    if skills:
        extra += " " + " ".join([f"SKILL_{s}" for s in skills])
    if yrs:
        extra += f" YEARS_{yrs}"
    return (cleaned + " " + extra).strip()

# Create processed X (text augmented with feature tokens)
X_raw = df[text_col].values
X = np.array([build_processed_text(t) for t in X_raw])
y = df[label_col].values

# -------------------------------------------------------------------------
# 4) Train-test split & pipeline
# -------------------------------------------------------------------------
stratify_param = y if len(np.unique(y)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)

pipeline = make_pipeline(
    TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words='english', min_df=2),
    LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='saga', random_state=42)
)

print("Training pipeline ...")
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, ROOT / "pipeline.pkl")
print("Saved pipeline.pkl to project root.")

# -------------------------------------------------------------------------
# 5) Evaluation & save metrics.json
# -------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

classes = None
try:
    classes = list(pipeline.named_steps['logisticregression'].classes_)
except Exception:
    try:
        classes = list(pipeline.classes_)
    except Exception:
        classes = None

metrics = {
    "accuracy": float(acc),
    "report": report,
    "confusion_matrix": cm,
    "classes": classes
}

with open(ROOT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics.json")
print("Done.")

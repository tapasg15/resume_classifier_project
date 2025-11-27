import re
def preprocess_text(t):
    if not t: return ""
    t = t.lower()
    t = re.sub(r'\n+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    # add any steps used in training (remove emails, phone nums)...
    return t.strip()

"""
ML Predictor Module
===================
Loads pre-trained models and provides clean prediction API for app.py.

Model 1 - Job Role: TF-IDF + Random Forest (42 categories, ~99.9%)
Model 2 - Score Grade: TF-IDF + Feature Engineering + Best Classifier (95-98%)
"""

import os
import re
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR   = os.path.join(BASE_DIR, "ml")

ROLE_MODEL_PATH  = os.path.join(ML_DIR, "job_role_model.pkl")
SCORE_MODEL_PATH = os.path.join(ML_DIR, "score_model.pkl")
LABEL_ENC_PATH   = os.path.join(ML_DIR, "label_encoder.pkl")
GRADE_ENC_PATH   = os.path.join(ML_DIR, "grade_encoder.pkl")
METRICS_PATH     = os.path.join(ML_DIR, "model_metrics.pkl")

GRADE_COLORS = {
    "Poor": "#f87171", "Average": "#fbbf24",
    "Good": "#a3e635", "Excellent": "#4ade80"
}

_role_model   = None
_score_bundle = None
_label_enc    = None
_metrics      = None


def _load_models():
    global _role_model, _score_bundle, _label_enc, _metrics
    if _role_model is None and os.path.exists(ROLE_MODEL_PATH):
        _role_model = joblib.load(ROLE_MODEL_PATH)
    if _score_bundle is None and os.path.exists(SCORE_MODEL_PATH):
        _score_bundle = joblib.load(SCORE_MODEL_PATH)
    if _label_enc is None and os.path.exists(LABEL_ENC_PATH):
        _label_enc = joblib.load(LABEL_ENC_PATH)
    if _metrics is None and os.path.exists(METRICS_PATH):
        _metrics = joblib.load(METRICS_PATH)


def is_ml_ready() -> bool:
    return os.path.exists(ROLE_MODEL_PATH) and os.path.exists(SCORE_MODEL_PATH)


def _extract_features(texts):
    """Extract numerical resume quality features (same logic as training)."""
    from scipy.sparse import csr_matrix
    features = []
    for text in texts:
        t = str(text).lower()
        years_exp = 0
        m = re.search(r'(\d+)\s*(?:\+\s*)?year[s]?', t)
        if m: years_exp = min(int(m.group(1)), 30)
        quant_count = len(re.findall(r'\d+%|\d+x\b|\d+\+', t))
        senior_words = ['senior', 'lead', 'principal', 'director', 'head of', 'manager',
                        'architect', 'vp ', 'chief', 'expert', 'specialist']
        senior_score = sum(1 for w in senior_words if w in t)
        cert_words = ['certified', 'certification', 'pmp', 'aws certified', 'google certified',
                      'microsoft certified', 'cissp', 'cfa', 'cpa', 'phd', 'doctorate',
                      'ccna', 'ccnp', 'azure', 'gcp', 'professional certificate']
        cert_count = sum(1 for w in cert_words if w in t)
        impact_verbs = ['reduced', 'increased', 'improved', 'delivered', 'led', 'built',
                        'designed', 'architected', 'launched', 'scaled', 'automated',
                        'optimized', 'deployed', 'developed', 'implemented', 'managed']
        impact_count = sum(1 for v in impact_verbs if v in t)
        weak_words = ['fresher', 'looking for', 'volunteer', 'intern', 'no experience',
                      'first job', 'seeking', 'basic knowledge', 'beginner', 'entry level',
                      'just graduated', 'recently completed', 'little experience']
        weak_score = sum(1 for w in weak_words if w in t)
        edu_score = 0
        if 'phd' in t or 'doctorate' in t: edu_score = 4
        elif "master" in t or "mba" in t or "m.s" in t or "m.tech" in t: edu_score = 3
        elif "bachelor" in t or "b.tech" in t or "b.e" in t or "b.sc" in t: edu_score = 2
        elif "diploma" in t or "associate" in t: edu_score = 1
        tech_terms = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes',
                      'react', 'node', 'django', 'machine learning', 'deep learning',
                      'tensorflow', 'pytorch', 'spark', 'kafka', 'mongodb', 'postgresql',
                      'ci/cd', 'git', 'api', 'microservices', 'cloud', 'devops']
        tech_density = sum(1 for t2 in tech_terms if t2 in t)
        top_companies = ['google', 'microsoft', 'amazon', 'apple', 'meta', 'netflix',
                         'ibm', 'oracle', 'salesforce', 'linkedin', 'twitter', 'uber',
                         'airbnb', 'stripe', 'mongodb']
        company_score = sum(1 for c in top_companies if c in t)
        text_len = min(len(t), 500) / 500.0
        features.append([years_exp, quant_count, senior_score, cert_count,
                          impact_count, weak_score, edu_score, tech_density,
                          company_score, text_len])
    return csr_matrix(np.array(features, dtype=float))


def predict_job_role(resume_text: str) -> dict:
    _load_models()
    if _role_model is None or _label_enc is None:
        return {"category": "Unknown", "confidence": 0.0, "top3": []}
    text = resume_text.lower().strip()
    if not text:
        return {"category": "Unknown", "confidence": 0.0, "top3": []}
    
    # Map vague ML category names to user-friendly descriptions
    FRIENDLY_NAMES = {
        'Technology': 'Software Development / IT',
        'Information Services': 'IT Services & Support',
        'Administration': 'Office Administration',
        'Conservation': 'Environmental Conservation',
        'Consulting': 'Management Consulting',
        'Private Sector Management': 'Business Management',
        'Skilled Trades': 'Skilled Trades & Technical Work',
        'Human Services & Social Work': 'Social Work & Counseling',
        'Media Production': 'Media & Content Production',
    }
    
    if hasattr(_role_model, "predict_proba"):
        proba = _role_model.predict_proba([text])[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top3 = []
        for i in top_idx:
            raw_name = str(_label_enc.classes_[i])
            friendly = FRIENDLY_NAMES.get(raw_name, raw_name)
            top3.append((friendly, round(float(proba[i]) * 100, 1)))
        raw_category = str(_label_enc.classes_[top_idx[0]])
        category   = FRIENDLY_NAMES.get(raw_category, raw_category)
        confidence = round(float(proba[top_idx[0]]) * 100, 1)
    else:
        pred = _role_model.predict([text])[0]
        raw_category = str(_label_enc.inverse_transform([pred])[0])
        category   = FRIENDLY_NAMES.get(raw_category, raw_category)
        confidence = 75.0
        top3       = [(category, confidence)]
    return {"category": category, "confidence": confidence, "top3": top3}


def predict_score(resume_text: str) -> dict:
    _load_models()
    if _score_bundle is None:
        return {"grade": "N/A", "grade_confidence": 0.0, "score": 0.0,
                "interpretation": "Model not loaded", "color": "#94a3b8"}
    text = resume_text.lower().strip()
    if not text:
        return {"grade": "N/A", "grade_confidence": 0.0, "score": 0.0,
                "interpretation": "Empty text", "color": "#94a3b8"}

    from scipy.sparse import hstack, csr_matrix

    # Load bundle components
    clf        = _score_bundle["classifier"]
    reg        = _score_bundle["regressor"]
    tfidf_clf  = _score_bundle["tfidf_clf"]
    tfidf_reg  = _score_bundle["tfidf_reg"]
    scaler_clf = _score_bundle["scaler_clf"]
    scaler_reg = _score_bundle["scaler_reg"]
    le_grade   = _score_bundle["label_enc"]

    # Build feature matrix for classification
    X_tfidf = tfidf_clf.transform([text])
    X_eng   = scaler_clf.transform(_extract_features([text]))
    X_combined = hstack([X_tfidf, X_eng])

    # Grade classification
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_combined)[0]
        best  = int(np.argmax(proba))
        grade = str(le_grade.classes_[best])
        grade_conf = round(float(proba[best]) * 100, 1)
    else:
        pred  = clf.predict(X_combined)[0]
        grade = str(le_grade.inverse_transform([pred])[0])
        grade_conf = 80.0

    # Numeric score from regressor
    Xr_tfidf = tfidf_reg.transform([text])
    Xr_eng   = scaler_reg.transform(_extract_features([text]))
    Xr_combined = hstack([Xr_tfidf, Xr_eng])
    raw_score = float(reg.predict(Xr_combined)[0])
    score = round(max(0.0, min(100.0, raw_score)), 1)

    interp_map = {
        "Excellent": "Highly competitive resume — strong candidate",
        "Good":      "Solid resume — competitive with minor improvements",
        "Average":   "Decent resume — notable improvements recommended",
        "Poor":      "Weak resume — significant restructuring needed"
    }
    return {
        "grade":             grade,
        "grade_confidence":  grade_conf,
        "score":             score,
        "interpretation":    interp_map.get(grade, ""),
        "color":             GRADE_COLORS.get(grade, "#94a3b8")
    }


def get_model_metrics() -> dict:
    _load_models()
    return _metrics if _metrics else {}

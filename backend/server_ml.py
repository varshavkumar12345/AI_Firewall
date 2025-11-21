# server_ml.py
# Backend ML server for Autonomous Firewall demo
# - Loads scaler.joblib, lstm.pt, xgb.joblib if present
# - Endpoint /predict_seq expects { history: [[pps,unique,syn], ...], current: [pps,unique,syn], ip: str }
# - Computes suspicious score via LSTM (if available) and decision via XGBoost (if available)
#
# Local uploaded model reference (for documentation):
# file:///mnt/data/trained_firewall_model.pkl

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from typing import List, Optional
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
import xgboost as xgb

# ---- CONFIG ----
MODEL_DIR = "."  # backend working dir
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm.pt")
XGB_PATH = os.path.join(MODEL_DIR, "xgb.joblib")
DEVICE = torch.device("cpu")

# ---- FLASK ----
app = Flask(__name__)
# CORS so frontend can talk to it
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173", "*"]}})

# ---- Pydantic request models ----
# Request payloads are parsed from JSON in Flask handlers (no pydantic validation here)

# ---- LSTM model class (must match training) ----
class SuspiciousLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last output
        return self.act(self.fc(last)).squeeze(1)

# ---- Load artifacts (guarded, fallback) ----
scaler = None
lstm = None
xgb_clf = None

def try_load():
    global scaler, lstm, xgb_clf
    # Load scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print("Failed to load scaler:", e)
            scaler = None
    else:
        scaler = None

    # Load LSTM
    if os.path.exists(LSTM_PATH):
        try:
            lstm = SuspiciousLSTM(input_dim=3, hidden=32, num_layers=1)
            lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
            lstm.to(DEVICE)
            lstm.eval()
        except Exception as e:
            print("Failed to load LSTM:", e)
            lstm = None
    else:
        lstm = None

    # Load XGBoost
    if os.path.exists(XGB_PATH):
        try:
            xgb_clf = joblib.load(XGB_PATH)
        except Exception as e:
            print("Failed to load xgb:", e)
            xgb_clf = None
    else:
        xgb_clf = None

try_load()

# ---- Utility functions ----
def compute_suspicious_score(history_np: np.ndarray) -> float:
    """
    history_np: shape (seq_len, 3)
    returns float in [0,1]
    Use LSTM if available, otherwise a simple heuristic.
    """
    # If no LSTM, fallback heuristic: normalized last pps and syn_ratio
    if lstm is None:
        pps = history_np[-1,0]
        syn = history_np[-1,2]
        s = min(1.0, (pps/2000)*0.6 + syn*0.6)
        return float(np.clip(s,0,1))

    # use LSTM (expects shape (1, seq_len, 3))
    arr = history_np.astype("float32")[None, ...]
    with torch.no_grad():
        t = torch.from_numpy(arr).to(DEVICE)
        out = lstm(t).cpu().numpy()
        return float(out[0])

def xgb_decision(feature_vec: np.ndarray) -> int:
    """
    feature_vec: shape (4,) ex: [pps, unique_ips, syn_ratio, suspicious]
    returns int class 0/1/2
    Uses predict_proba with thresholds for demonstrative behavior.
    """
    # fallback rule-based decision if xgb is missing
    if xgb_clf is None:
        pps, uniq, syn, susp = feature_vec
        if pps > 1500 or susp > 0.8 or syn > 0.7:
            return 2
        if pps > 800 or susp > 0.5:
            return 1
        return 0

    X = feature_vec.reshape(1, -1)
    try:
        probs = xgb_clf.predict_proba(X)[0]
        # probs ordering depends on training but usually [class0, class1, class2]
        # guard: if shape mismatch, fallback to predict
        if len(probs) < 3:
            return int(xgb_clf.predict(X)[0])
        prob_allow, prob_rate, prob_block = probs[0], probs[1], probs[2]
    except Exception:
        # fallback to predict if predict_proba fails
        return int(xgb_clf.predict(X)[0])

    # tuned demo thresholds (adjust to taste)
    if prob_block >= 0.45 or prob_block > max(prob_allow, prob_rate) * 1.2:
        return 2
    if prob_rate >= 0.40 or prob_rate > max(prob_allow, prob_block) * 1.1:
        return 1
    return 0

# ---- Endpoints ----
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "models": {"scaler": os.path.exists(SCALER_PATH), "lstm": os.path.exists(LSTM_PATH), "xgb": os.path.exists(XGB_PATH)}})

@app.route("/debug_info", methods=["GET"])
def debug_info():
    return jsonify({
        "scaler_exists": os.path.exists(SCALER_PATH),
        "lstm_exists": os.path.exists(LSTM_PATH),
        "xgb_exists": os.path.exists(XGB_PATH),
        "scaler_n_features": getattr(scaler, "n_features_in_", None)
    })

@app.route("/predict_seq", methods=["POST"])
def predict_seq():
    try:
        body = request.get_json(force=True)
        if body is None:
            return make_response(jsonify({"detail": "Invalid JSON payload"}), 400)

        history = body.get("history")
        current = body.get("current")
        ip = body.get("ip")

        # validate lengths
        if not isinstance(current, (list, tuple)) or len(current) != 3:
            return make_response(jsonify({"detail": "current must be length 3: [pps, unique_ips, syn_ratio]"}), 400)
        if not isinstance(history, (list, tuple)) or len(history) < 2:
            return make_response(jsonify({"detail": "history must be at least length 2 (seq of [pps,unique_ips,syn_ratio])"}), 400)

        # raw arrays
        hist = np.array(history, dtype=np.float32)  # (seq_len, 3)
        cur = np.array(current, dtype=np.float32).reshape(1, -1)  # (1,3)

        # ---- SCALE / PREPARE FEATURES SAFELY ----
        # Compute suspicious using the (raw) history (our LSTM was trained on raw sequences)
        hist_raw = hist.astype(np.float32)
        susp_score = compute_suspicious_score(hist_raw)

        # Build the 4-feature vector that XGBoost expects: [pps, unique_ips, syn_ratio, suspicious]
        cur_raw = cur.flatten().astype(np.float32)  # shape (3,)
        feat_raw = np.concatenate([cur_raw, np.array([susp_score], dtype=np.float32)])  # shape (4,)

        # If scaler exists and expects 4 features, scale this 4-feature vector.
        if scaler is not None:
            n_in = getattr(scaler, "n_features_in_", None)
            try:
                if n_in is None:
                    # try transform, catch errors
                    feat_scaled = scaler.transform(feat_raw.reshape(1, -1))[0]
                else:
                    if n_in == feat_raw.shape[0]:
                        feat_scaled = scaler.transform(feat_raw.reshape(1, -1))[0]
                    elif n_in == cur_raw.shape[0]:
                        # scaler was fitted on 3 features (no suspicious)
                        cur_scaled = scaler.transform(cur_raw.reshape(1, -1))[0]
                        feat_scaled = np.concatenate([cur_scaled, np.array([susp_score], dtype=np.float32)])
                    else:
                        # unexpected scaler shape — fallback to raw
                        feat_scaled = feat_raw
            except Exception as e:
                # if scaling fails, log and fallback to raw features
                print("Scaler transform failed:", e)
                feat_scaled = feat_raw
        else:
            feat_scaled = feat_raw

        feat = feat_scaled.astype(np.float32)

        # compute decision
        action = xgb_decision(feat)  # 0/1/2

        return jsonify({"action": int(action), "suspicious": float(susp_score), "ip": ip})
    except Exception as e:
        # For development: return error detail instead of generic 500 (helps debugging)
        # NOTE: remove or change for production
        return make_response(jsonify({"error": str(e), "type": type(e).__name__}), 500)

# ---- reload endpoint to reload models without restarting server (helpful) ----
@app.route("/reload_models", methods=["POST"])
def reload_models():
    try:
        try_load()
        return jsonify({"reloaded": True})
    except Exception as e:
        return make_response(jsonify({"detail": str(e)}), 500)


if __name__ == "__main__":
    # default Flask dev server for local testing
    app.run(host="0.0.0.0", port=8000, debug=True)

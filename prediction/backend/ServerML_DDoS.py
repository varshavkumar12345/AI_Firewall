from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
import xgboost as xgb
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
SCALER_PATH= os.getenv("SCALER_PATH")
LSTM_PATH= os.getenv("LSTM_PATH")
XGB_PATH= os.getenv("XGB_PATH")
INTERMEDIATE_PATH= os.getenv("INTERMEDIATE_JSON_PATH")
DEVICE= torch.device("cpu")

# Confidence threshold for human review (if prediction confidence is below this, flag for review)
CONFIDENCE_THRESHOLD = 0.65

# ---- FLASK APP ----
app = Flask(__name__)
CORS(app)

# ---- LSTM model class ----
class SuspiciousLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.act(self.fc(last)).squeeze(1)

# ---- Load artifacts ----
scaler = None
lstm = None
xgb_clf = None

def try_load():
    global scaler, lstm, xgb_clf
    
    # Load scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"✓ Loaded scaler from {SCALER_PATH}")
        except Exception as e:
            print(f"✗ Failed to load scaler: {e}")
            scaler = None
    else:
        print(f"✗ Scaler not found at {SCALER_PATH}")
        scaler = None

    # Load LSTM
    if os.path.exists(LSTM_PATH):
        try:
            lstm = SuspiciousLSTM(input_dim=3, hidden=32, num_layers=1)
            lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
            lstm.to(DEVICE)
            lstm.eval()
            print(f"✓ Loaded LSTM from {LSTM_PATH}")
        except Exception as e:
            print(f"✗ Failed to load LSTM: {e}")
            lstm = None
    else:
        print(f"✗ LSTM not found at {LSTM_PATH}")
        lstm = None

    # Load XGBoost
    if os.path.exists(XGB_PATH):
        try:
            xgb_clf = joblib.load(XGB_PATH)
            print(f"✓ Loaded XGBoost from {XGB_PATH}")
        except Exception as e:
            print(f"✗ Failed to load XGBoost: {e}")
            xgb_clf = None
    else:
        print(f"✗ XGBoost not found at {XGB_PATH}")
        xgb_clf = None

try_load()

# ---- Utility functions ----
def compute_suspicious_score(history_np: np.ndarray) -> float:
    """Compute suspicious score using LSTM or fallback heuristic"""
    if lstm is None:
        pps = history_np[-1, 0]
        syn = history_np[-1, 2]
        s = min(1.0, (pps/2000)*0.6 + syn*0.6)
        return float(np.clip(s, 0, 1))
    
    arr = history_np.astype("float32")[None, ...]
    with torch.no_grad():
        t = torch.from_numpy(arr).to(DEVICE)
        out = lstm(t).cpu().numpy()
        return float(out[0])

def xgb_decision(feature_vec: np.ndarray) -> tuple:
    """
    Make decision using XGBoost
    Returns: (action: int, confidence: float)
    """
    if xgb_clf is None:
        # Fallback rule-based decision
        pps, uniq, syn, susp = feature_vec
        if pps > 1500 or susp > 0.8 or syn > 0.7:
            return 2, 0.85
        if pps > 800 or susp > 0.5:
            return 1, 0.70
        return 0, 0.90
    
    X = feature_vec.reshape(1, -1)
    try:
        probs = xgb_clf.predict_proba(X)[0]
        if len(probs) < 3:
            action = int(xgb_clf.predict(X)[0])
            return action, 0.70
        
        prob_allow, prob_rate, prob_block = probs[0], probs[1], probs[2]
        
        # Determine action with tuned thresholds
        if prob_block >= 0.45 or prob_block > max(prob_allow, prob_rate) * 1.2:
            action = 2
        elif prob_rate >= 0.40 or prob_rate > max(prob_allow, prob_block) * 1.1:
            action = 1
        else:
            action = 0
        
        # Confidence is the probability of the chosen action
        confidence = float(probs[action])
        
        return action, confidence
    
    except Exception as e:
        print(f"XGBoost prediction error: {e}")
        action = int(xgb_clf.predict(X)[0])
        return action, 0.70

def log_to_intermediate(history, current, ip, action, suspicious, confidence):
    """Log prediction to intermediate file for potential human review"""
    try:
        # Flag for human review if confidence is low
        needs_review = confidence < CONFIDENCE_THRESHOLD
        
        entry = {
            "history": history,
            "current": current,
            "ip": ip,
            "predicted_action": int(action),
            "predicted_suspicious": float(suspicious),
            "confidence": float(confidence),
            "needs_review": needs_review,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Load existing data
        if os.path.exists(INTERMEDIATE_PATH):
            with open(INTERMEDIATE_PATH, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        else:
            data = []
        
        # Append new entry
        data.append(entry)
        
        # Keep only last 1000 entries to prevent file from growing too large
        if len(data) > 1000:
            data = data[-1000:]
        
        # Save back
        with open(INTERMEDIATE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        
        if needs_review:
            print(f"⚠ Low confidence prediction logged for review: IP={ip}, Action={action}, Confidence={confidence:.2f}")
    
    except Exception as e:
        print(f"Error logging to intermediate file: {e}")

# ---- Endpoints ----
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "scaler": os.path.exists(SCALER_PATH),
            "lstm": os.path.exists(LSTM_PATH),
            "xgb": os.path.exists(XGB_PATH)
        },
        "human_review_enabled": True,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

@app.route('/debug_info', methods=['GET'])
def debug_info():
    return jsonify({
        "scaler_exists": os.path.exists(SCALER_PATH),
        "lstm_exists": os.path.exists(LSTM_PATH),
        "xgb_exists": os.path.exists(XGB_PATH),
        "scaler_n_features": getattr(scaler, "n_features_in_", None),
        "intermediate_path": INTERMEDIATE_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

@app.route('/predict_seq', methods=['POST'])
def predict_seq():
    try:
        body = request.get_json()
        
        if not body:
            return jsonify({"error": "No JSON data provided"}), 400
        
        history = body.get('history')
        current = body.get('current')
        ip = body.get('ip')
        
        # Validate inputs
        if not current or len(current) != 3:
            return jsonify({
                "error": "current must be length 3: [pps, unique_ips, syn_ratio]"
            }), 400
        
        if not history or len(history) < 2:
            return jsonify({
                "error": "history must be at least length 2"
            }), 400
        
        # Convert to numpy arrays
        hist = np.array(history, dtype=np.float32)
        cur = np.array(current, dtype=np.float32).reshape(1, -1)
        
        # Compute suspicious score
        hist_raw = hist.astype(np.float32)
        susp_score = compute_suspicious_score(hist_raw)
        
        # Build feature vector [pps, unique_ips, syn_ratio, suspicious]
        cur_raw = cur.flatten().astype(np.float32)
        feat_raw = np.concatenate([cur_raw, np.array([susp_score], dtype=np.float32)])
        
        # Scale features if scaler exists
        if scaler is not None:
            n_in = getattr(scaler, "n_features_in_", None)
            try:
                if n_in is None or n_in == feat_raw.shape[0]:
                    feat_scaled = scaler.transform(feat_raw.reshape(1, -1))[0]
                elif n_in == cur_raw.shape[0]:
                    cur_scaled = scaler.transform(cur_raw.reshape(1, -1))[0]
                    feat_scaled = np.concatenate([cur_scaled, np.array([susp_score], dtype=np.float32)])
                else:
                    feat_scaled = feat_raw
            except Exception as e:
                print(f"Scaler transform failed: {e}")
                feat_scaled = feat_raw
        else:
            feat_scaled = feat_raw
        
        feat = feat_scaled.astype(np.float32)
        
        # Make decision
        action, confidence = xgb_decision(feat)
        
        # Log to intermediate file for potential human review
        log_to_intermediate(
            history,
            current,
            ip or "unknown",
            action,
            susp_score,
            confidence
        )
        
        return jsonify({
            "action": int(action),
            "suspicious": float(susp_score),
            "confidence": float(confidence),
            "ip": ip,
            "needs_review": confidence < CONFIDENCE_THRESHOLD
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Reload models without restarting server"""
    try:
        try_load()
        return jsonify({"reloaded": True, "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/intermediate_stats', methods=['GET'])
def intermediate_stats():
    """Get statistics about intermediate predictions"""
    try:
        if not os.path.exists(INTERMEDIATE_PATH):
            return jsonify({
                "total_predictions": 0,
                "needs_review": 0,
                "review_percentage": 0.0
            })
        
        with open(INTERMEDIATE_PATH, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = []
        
        total = len(data)
        needs_review = sum(1 for item in data if item.get('needs_review', False))
        review_pct = (needs_review / total * 100) if total > 0 else 0.0
        
        return jsonify({
            "total_predictions": total,
            "needs_review": needs_review,
            "review_percentage": round(review_pct, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Startup message
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DDoS AUTONOMOUS FIREWALL ML SERVER (Flask)")
    print("=" * 80)
    print("LSTM + XGBoost Prediction Pipeline")
    print("Human-in-the-Loop Continuous Learning Enabled")
    print("=" * 80)
    print(f"\nConfidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Intermediate Log: {INTERMEDIATE_PATH}")
    print("\nEndpoints:")
    print("   POST /predict_seq          - Make DDoS prediction")
    print("   POST /reload_models        - Reload trained models")
    print("   GET  /health               - Health check")
    print("   GET  /debug_info           - Debug information")
    print("   GET  /intermediate_stats   - Review statistics")
    print("=" * 80 + "\n")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
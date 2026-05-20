import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── FLOW TABLE ───────────────────────────────────────────
# This is your SDN contribution — clinician-adjustable rules
# Change these thresholds without touching the ML model
FLOW_TABLE = {
    "SAFE":    (0.0, 0.4),   # R1: risk < 0.4
    "CAUTION": (0.4, 0.7),   # R2: 0.4 ≤ risk < 0.7
    "ALERT":   (0.7, 1.0),   # R3: risk ≥ 0.7
}

WINDOW_SIZE = {
    "SAFE":    30,   # seconds — normal monitoring
    "CAUTION": 10,   # seconds — QoS adaptation
    "ALERT":   10,   # seconds — maximum resolution
}

ESCALATION_THRESHOLD = 3  # R4: consecutive ALERTs before escalation

# ── CONTROLLER CLASS ─────────────────────────────────────
class SDNController:
    def __init__(self):
        self.alert_log       = []        # northbound API log
        self.consecutive_alerts = 0      # R4 counter
        self.current_window  = 30        # current QoS window size
        self.model           = None
        self.scaler          = None

    def load_model(self, X, y):
        """Train and store the Logistic Regression model."""
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)
        self.model  = LogisticRegression(
            class_weight="balanced", max_iter=1000
        )
        self.model.fit(X_scaled, y)
        print("✅ SDN Controller: model loaded")

    def apply_flow_table(self, risk_score):
        """
        Core SDN function — apply flow table rules.
        Takes risk score → returns status + window size.
        This is your control plane logic.
        """
        for status, (low, high) in FLOW_TABLE.items():
            if low <= risk_score < high:
                return status
        return "ALERT"  # fallback

    def process_window(self, features, timestamp):
        """
        Process one EEG window through the full SDN pipeline.
        features: 1D numpy array (69 features)
        timestamp: float (seconds into recording)
        """
        # Scale features
        features_scaled = self.scaler.transform(
            features.reshape(1, -1)
        )

        # Get risk score from model (control plane)
        risk_score = self.model.predict_proba(
            features_scaled
        )[0][1]

        # Apply flow table (SDN controller decision)
        status = self.apply_flow_table(risk_score)

        # QoS window adaptation
        self.current_window = WINDOW_SIZE[status]

        # R4 — escalation logic
        if status == "ALERT":
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = 0

        if self.consecutive_alerts >= ESCALATION_THRESHOLD:
            status = "ESCALATE"

        # Log to northbound API
        entry = {
            "timestamp": timestamp,
            "risk_score": round(risk_score, 3),
            "status":     status,
            "window_size": self.current_window,
        }
        self.alert_log.append(entry)

        return entry

    def get_alert_history(self):
        """Return all non-SAFE events — northbound API output."""
        return [e for e in self.alert_log if e["status"] != "SAFE"]

    def print_status(self, entry):
        """Print one window result to terminal."""
        icons = {
            "SAFE":     "🟢",
            "CAUTION":  "🟡",
            "ALERT":    "🔴",
            "ESCALATE": "🚨",
        }
        icon = icons.get(entry["status"], "⚪")
        print(
            f"[{entry['timestamp']:>8.1f}s] "
            f"{icon} {entry['status']:<10} "
            f"risk={entry['risk_score']:.3f}  "
            f"window={entry['window_size']}s"
        )


# ── TEST THE CONTROLLER ──────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Load data
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    # Initialise and train controller
    controller = SDNController()
    controller.load_model(X, y)

    print("\n── Running SDN Controller on chb01_03 ──\n")

    # Simulate processing windows from chb01_03
    # Seizure is at seconds 2996-3036
    # With 30s windows and 50% overlap, window starts at:
    # 2996 - 30 = ~2966s onward
    print(f"{'Timestamp':>12} {'Status':<12} {'Risk':>6}  {'Window':>8}")
    print("─" * 50)

    # Process all windows
    sample_rate   = 256
    window_size   = 30
    step          = 15  # 50% overlap

    # Load raw file to get actual features
    # Use last 30 windows around seizure onset (2996s)
    seizure_onset = 2996
    start_window  = max(0, seizure_onset - 300)  # 5 mins before

    for i, (features, label) in enumerate(zip(X, y)):
        timestamp = i * step  # approximate timestamp
        entry = controller.process_window(features, timestamp)
        controller.print_status(entry)

    # Summary
    print("\n── Alert History (non-SAFE events) ──\n")
    history = controller.get_alert_history()
    if history:
        for e in history:
            print(f"  [{e['timestamp']:>8.1f}s] "
                  f"{e['status']:<10} "
                  f"risk={e['risk_score']}")
    else:
        print("  No alerts triggered.")

    print(f"\nTotal windows processed: {len(controller.alert_log)}")
    print(f"Alerts triggered:        "
          f"{sum(1 for e in controller.alert_log if e['status'] in ['ALERT','ESCALATE'])}")
    print(f"CAUTION events:          "
          f"{sum(1 for e in controller.alert_log if e['status'] == 'CAUTION')}")
import mne
import numpy as np
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────
WINDOW_SIZE_DEFAULT = 30   # seconds (interictal / normal)
WINDOW_SIZE_QOS     = 10   # seconds (CAUTION state — QoS adaptation)
OVERLAP             = 0.5  # 50% overlap between windows
SAMPLE_RATE         = 256  # Hz — from CHB-MIT summary

# Seizure annotations for files we have
SEIZURE_ANNOTATIONS = {
    "chb01_03.edf": [(2996, 3036)],
    "chb01_04.edf": [(1467, 1494)],
    "chb01_15.edf": [(1732, 1772)],
    "chb01_16.edf": [(1015, 1066)],
    "chb01_18.edf": [(1720, 1810)],
    "chb01_21.edf": [(327,  420)],
    "chb01_26.edf": [(1862, 1963)],
}

EDF_FILES = [
    "data/chb01/chb01_01.edf",   # baseline, no seizure
    "data/chb01/chb01_03.edf",
    "data/chb01/chb01_04.edf",
    "data/chb01/chb01_15.edf",
    "data/chb01/chb01_16.edf",
    "data/chb01/chb01_18.edf",
    "data/chb01/chb01_21.edf",
    "data/chb01/chb01_26.edf",
]

def load_edf(filepath):
    """Load an EDF file and return raw MNE object."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    return raw

def get_windows(raw, window_size=WINDOW_SIZE_DEFAULT):
    """
    Slice raw EEG into windows of window_size seconds with 50% overlap.
    Returns list of (start_sample, end_sample) tuples.
    """
    n_samples = raw.n_times
    win_samples = int(window_size * SAMPLE_RATE)
    step = int(win_samples * (1 - OVERLAP))
    windows = []
    start = 0
    while start + win_samples <= n_samples:
        windows.append((start, start + win_samples))
        start += step
    return windows

def label_window(start_sample, end_sample, seizure_times):
    """
    Label a window as ictal (1) or interictal (0).
    seizure_times: list of (onset_sec, offset_sec) tuples.
    """
    start_sec = start_sample / SAMPLE_RATE
    end_sec   = end_sample   / SAMPLE_RATE
    for onset, offset in seizure_times:
        if start_sec < offset and end_sec > onset:
            return 1  # overlaps seizure
    return 0

def extract_features(window_data):
    """
    Extract 3 features per channel from a window.
    window_data: numpy array of shape (n_channels, n_samples)
    Returns: 1D feature vector (n_channels * 3,)
    """
    features = []
    for ch in range(window_data.shape[0]):
        signal = window_data[ch]

        # Feature 1 — Variance
        variance = np.var(signal)

        # Feature 2 — Line Length
        line_length = np.sum(np.abs(np.diff(signal)))

        # Feature 3 — Band Power (beta: 12-30 Hz, most relevant for seizures)
        fft_vals  = np.abs(np.fft.rfft(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/SAMPLE_RATE)
        beta_mask = (fft_freqs >= 12) & (fft_freqs <= 30)
        band_power = np.mean(fft_vals[beta_mask] ** 2)

        features.extend([variance, line_length, band_power])

    return np.array(features)

def process_file(filepath, seizure_times, window_size=WINDOW_SIZE_DEFAULT):
    """
    Full pipeline for one EDF file.
    Returns: X (features), y (labels)
    """
    print(f"Processing {filepath}...")
    raw      = load_edf(filepath)
    data     = raw.get_data()  # shape: (n_channels, n_samples)
    windows  = get_windows(raw, window_size)

    X, y = [], []
    for start, end in windows:
        window_data = data[:, start:end]
        features    = extract_features(window_data)
        label       = label_window(start, end, seizure_times)
        X.append(features)
        y.append(label)

    print(f"  → {len(windows)} windows, {sum(y)} ictal, {len(y)-sum(y)} interictal")
    return np.array(X), np.array(y)

# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    all_X, all_y = [], []

    for filename, seizure_times in SEIZURE_ANNOTATIONS.items():
        filepath = f"data/chb01/{filename}"
        X, y = process_file(filepath, seizure_times)
        all_X.append(X)
        all_y.append(y)

    # Add baseline file (no seizures)
    X_base, y_base = process_file("data/chb01/chb01_01.edf", seizure_times=[])
    all_X.append(X_base)
    all_y.append(y_base)

    # Combine all files
    X_final = np.vstack(all_X)
    y_final = np.concatenate(all_y)

    print(f"\nFinal dataset: {X_final.shape[0]} windows, {X_final.shape[1]} features each")
    print(f"Ictal windows: {sum(y_final)}")
    print(f"Interictal windows: {len(y_final) - sum(y_final)}")

    # Save
    np.save("data/X.npy", X_final)
    np.save("data/y.npy", y_final)
    print("\nSaved to data/X.npy and data/y.npy")
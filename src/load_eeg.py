import mne

raw = mne.io.read_raw_edf("data/chb01/chb01_01.edf", preload=True)
print(raw.info)
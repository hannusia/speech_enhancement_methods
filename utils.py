import numpy as np
import librosa


def snr_db(clean, noisy):
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


def resample_signal(signal, orig_sr, target_sr=16000):
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)

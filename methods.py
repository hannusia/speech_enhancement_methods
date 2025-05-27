import numpy as np
import scipy.signal
import scipy.fft


def wiener_filter(noisy_signal, noise_estimate, frame_size=1024, overlap=512):
    f, t, Zxx_noisy = scipy.signal.stft(
        noisy_signal, nperseg=frame_size, noverlap=overlap)
    _, _, Zxx_noise = scipy.signal.stft(
        noise_estimate, nperseg=frame_size, noverlap=overlap)
    noise_psd = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)
    signal_psd = np.abs(Zxx_noisy) ** 2
    gain = signal_psd / (signal_psd + noise_psd)
    enhanced_spectrogram = gain * Zxx_noisy
    _, enhanced_signal = scipy.signal.istft(
        enhanced_spectrogram, nperseg=frame_size, noverlap=overlap)
    return np.real(enhanced_signal[:len(noisy_signal)])


def spectral_subtraction(signal, noise_estimate, frame_size=1024, overlap=512, noise_floor=0.01):
    window = scipy.signal.windows.hamming(frame_size)
    noise_spectrum = np.mean([
        np.abs(scipy.fft.fft(noise_estimate[i:i+frame_size] * window))
        for i in range(0, len(noise_estimate) - frame_size, overlap)
    ], axis=0)
    hop_size = frame_size - overlap
    n_frames = int(np.ceil(len(signal) / hop_size))
    padded_length = n_frames * hop_size + (frame_size - hop_size)
    padded_signal = np.pad(signal, (0, padded_length - len(signal)))
    processed_signal = np.zeros_like(padded_signal)
    weight_sum = np.zeros_like(padded_signal)
    window = scipy.signal.windows.hamming(frame_size)
    for i in range(n_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        frame = padded_signal[start_idx:end_idx]
        windowed_frame = frame * window
        frame_spectrum = scipy.fft.fft(windowed_frame)
        subtracted_spectrum = np.abs(frame_spectrum) - noise_spectrum
        subtracted_spectrum = np.maximum(
            subtracted_spectrum, noise_floor * np.max(noise_spectrum))
        processed_frame = scipy.fft.ifft(
            subtracted_spectrum * np.exp(1j * np.angle(frame_spectrum)))
        processed_frame = np.real(processed_frame)
        processed_signal[start_idx:end_idx] += processed_frame * window
        weight_sum[start_idx:end_idx] += window**2
    weight_sum[weight_sum == 0] = 1
    processed_signal /= weight_sum
    return processed_signal[:len(signal)]

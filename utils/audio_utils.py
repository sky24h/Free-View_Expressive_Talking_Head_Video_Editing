import librosa
import librosa.filters
import numpy as np

# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile

hp_num_mels = 80
hp_rescale = True
hp_rescaling_max = 0.9
hp_use_lws = False
hp_n_fft = 800
hp_hop_size = 200
hp_win_size = 800
hp_sample_rate = 16000
hp_frame_shift_ms = None
hp_signal_normalization = True
hp_allow_clipping_in_normalization = True
hp_symmetric_mels = True
hp_max_abs_value = 4.0
hp_preemphasize = True
hp_preemphasis = 0.97
hp_min_level_db = -100
hp_ref_level_db = 20
hp_fmin = 55
hp_fmax = 7600


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp_hop_size
    if hop_size is None:
        assert hp_frame_shift_ms is not None
        hop_size = int(hp_frame_shift_ms / 1000 * hp_sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp_preemphasis, hp_preemphasize))
    S = _amp_to_db(np.abs(D)) - hp_ref_level_db
    if hp_signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp_preemphasis, hp_preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp_ref_level_db
    if hp_signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    import lws

    return lws.lws(hp_n_fft, get_hop_size(), fftsize=hp_win_size, mode="speech")


def _stft(y):
    if hp_use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp_n_fft, hop_length=get_hop_size(), win_length=hp_win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram"""
    pad = fsize - fshift
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding"""
    M = num_frames(len(x), fsize, fshift)
    pad = fsize - fshift
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp_fmax <= hp_sample_rate // 2
    return librosa.filters.mel(hp_sample_rate, hp_n_fft, n_mels=hp_num_mels, fmin=hp_fmin, fmax=hp_fmax)


def _amp_to_db(x):
    min_level = np.exp(hp_min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    if hp_allow_clipping_in_normalization:
        if hp_symmetric_mels:
            return np.clip(
                (2 * hp_max_abs_value) * ((S - hp_min_level_db) / (-hp_min_level_db)) - hp_max_abs_value,
                -hp_max_abs_value,
                hp_max_abs_value,
            )
        else:
            return np.clip(
                hp_max_abs_value * ((S - hp_min_level_db) / (-hp_min_level_db)),
                0,
                hp_max_abs_value,
            )

    assert S.max() <= 0 and S.min() - hp_min_level_db >= 0
    if hp_symmetric_mels:
        return (2 * hp_max_abs_value) * ((S - hp_min_level_db) / (-hp_min_level_db)) - hp_max_abs_value
    else:
        return hp_max_abs_value * ((S - hp_min_level_db) / (-hp_min_level_db))
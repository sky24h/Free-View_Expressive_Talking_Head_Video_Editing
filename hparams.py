from glob import glob
import os

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hparams = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,

    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    ###################### Our training parameters #################################
    img_size=256,
    fps=25,
    log_interval=1000, # steps
    num_workers=16,
    save_optimizer_state=True,

    syncnet_batch_size            = 256,
    syncnet_lr                    = 1e-4,
    syncnet_nepochs               = 50000,
    syncnet_save_after_nepochs    = 1000,
    syncnet_not_improved_limit    = 1000,  # epochs
    syncnet_eval_interval         = 10000, # steps
    syncnet_initial_save_interval = 10000, # steps

    generator_batch_size            = 48,
    generator_lr                    = 1e-4,
    generator_nepochs               = 5000,
    generator_save_after_nepochs    = 10,
    generator_initial_save_interval = 2000,  # steps
    generator_eval_interval         = 2000,  # steps
    generator_sample_interval       = 10000, # steps
    generator_syncnet_wt            = 0.2,   # is initially zero, will be set automatically to 0.05 later. Leads to faster convergence.
    vgg_wt                          = 0.2,
    blink_wt                        = 0.2,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)

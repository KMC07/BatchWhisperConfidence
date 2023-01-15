import os
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def load_audio_waveform_img(audio: Union[str, bytes, np.ndarray, torch.Tensor],
                            h: int, w: int, ignore_shift=False) -> np.ndarray:
    """

    Parameters
    ----------
    audio: Union[str, bytes, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or bytes of audio file or a NumPy array or Tensor containing the audio waveform in 16 kHz
    h: int
        Height of waveform image
    w: int
        Width of waveform image
    ignore_shift: bool
        ignore warning if NumpPy array or PyTorch Tensor is used for audio

    Returns
    -------
    Audio waveform image as a NumPy array, in uint8 dtype.
    """

    try:
        if isinstance(audio, str):
            stream = ffmpeg.input(audio, threads=0)
            inp = None

        else:
            if isinstance(audio, bytes):
                stream = ffmpeg.input('pipe:', threads=0)
                inp = audio
            else:
                if not ignore_shift:
                    warnings.warn('A resampled input causes an unexplained temporal shift in waveform image '
                                  'that will skew the timestamp suppression and may result in inaccurate timestamps.\n'
                                  'Use audio_for_mask for transcribe() to provide the original audio track '
                                  'as the path or bytes of the audio file.',
                                  stacklevel=2)
                stream = ffmpeg.input('pipe:', threads=0, ac=1, format='s16le')
                if isinstance(audio, torch.Tensor):
                    audio = np.array(audio)
                inp = (audio * 32768.0).astype(np.int16).tobytes()

        waveform, err = (
            stream.filter('aformat', channel_layouts='mono')
            .filter('highpass', f='200').filter('lowpass', f='3000')
            .filter('showwavespic', s=f'{w}x{h}')
            .output('-', pix_fmt='gray', format='rawvideo', vframes=1)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio in waveform: {e.stderr.decode()}") from e
    else:
        if not waveform:
            partial_file = b'partial file' in err and b'Output file is empty' in err
            add_msg = '\nMetadata for decoding are likely at end of file, try to use path of audio instead.' \
                if partial_file and isinstance(audio, bytes) else ''
            raise RuntimeError(f"Failed to load audio in waveform: {err.decode()}" + add_msg)
        return np.frombuffer(waveform, dtype=np.uint8).reshape(h, w)


def remove_lower_quantile(waveform_img: np.ndarray,
                          upper_quantile: float = None,
                          lower_quantile: float = None,
                          lower_threshold: float = None) -> np.ndarray:
    """
    Removes lower quantile of amplitude from waveform image
    """
    if upper_quantile is None:
        upper_quantile = 0.85
    if lower_quantile is None:
        lower_quantile = 0.15
    if lower_threshold is None:
        lower_threshold = 0.15
    waveform_img = deepcopy(waveform_img)
    wave_sums = waveform_img.sum(0)
    mx = np.quantile(wave_sums, upper_quantile, -1)
    mn = np.quantile(wave_sums, lower_quantile, -1)
    mn_threshold = (mx - mn) * lower_threshold + mn
    waveform_img[:, wave_sums < mn_threshold] = 0
    return waveform_img


def wave_to_ts_filter(waveform_img: np.ndarray, suppress_middle=True,
                      max_index: (list, int) = None) -> np.ndarray:
    """
    Returns A NumPy array mask of sections with amplitude zero
    """
    assert waveform_img.ndim <= 2, f'waveform have at most 2 dims but found {waveform_img.ndim}'
    if waveform_img.ndim == 1:
        wave_sum = waveform_img
    else:
        wave_sum = waveform_img.sum(-2)

    wave_filter = wave_sum.astype(bool)

    if not suppress_middle:
        nonzero_indices = wave_filter.nonzero()[0]
        wave_filter[nonzero_indices[0]:nonzero_indices[-1] + 1] = True
    if max_index is not None:
        wave_filter[max_index + 1:] = False

    return ~wave_filter

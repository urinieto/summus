"""Some util functions for summus."""

import numpy as np


class Segment:
    def __init__(self, start_time, end_time, sr):
        """Start and end times in seconds."""
        self.start_time = start_time
        self.end_time = end_time
        self.sr = sr
        self.start_sample = time_to_sample(start_time, sr)
        self.end_sample = time_to_sample(end_time, sr)
        self.n_samples = self.end_sample - self.start_sample


def time_to_sample(time, sr):
    """Computes the sample corresponding to the current time.

    Parameters
    ----------
    time : float
        Time in seconds
    sr : sampling rate
        Sampling rate to convert the time.

    Returns
    -------
    sample : int
        Time in sample.
    """
    return int(time * sr)


def f_measure(precision, recall):
    """Computes the harmonic mean between precision and recall.

    Parameters
    ----------
    precision : float > 0 < 1
        Precision value.
    recall : float > 0 < 1
        Recall value.

    Returns
    -------
    f_measure : float > 0 < 1
        Harmonic mean between precision and recall.
    """
    return 2 * precision * recall / (precision + recall)


def generate_fade_audio(fade_seg, audio, is_out):
    """Generates the fade in or out audio.

    Parameters
    ----------
    fade_seq : Segment
        Segment containing the start and ending points.
    audio : np.array
        The actual audio from which to generate the fade in/out.
    is_out : bool
        Whether to do a fade out (True) or fade in (False).

    Returns
    -------
    fade : np.array
        The generated fade in/out audio.
    """
    mask = np.arange(fade_seg.n_samples) / float(fade_seg.n_samples)
    if is_out:
        mask = 1 - mask
    return audio[fade_seg.start_sample:fade_seg.end_sample] * mask

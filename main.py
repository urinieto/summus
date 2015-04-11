#!/usr/bin/env python
"""
This script computes the music summary of an audio file.

Based on:
    1. Nieto O, Humphrey EJ, Bello JP. Compressing Audio Recordings into Music
    Summaries. In: Proc. of the 13th International Society for Music
    Information Retrieval Conference. Porto, Portugal; 2012:313-318.

Examples:
    TODO
"""
import argparse
import logging
import numpy as np
import os
import time

import librosa

# Analysis Globals
FFT_SIZE = 2048
HOP_SIZE = 512
SAMPLING_RATE = 22050
N_MELS = 128
N_MFCCS = 14

# Feature Globals
PCP_TYPE = "pcp"
TONNETZ_TYPE = "tonnetz"
MFCC_TYPE = "mfcc"
FEATURE_TYPES = [PCP_TYPE, TONNETZ_TYPE, MFCC_TYPE]


def compute_beats(y_percussive, sr=22050):
    """Computes the beats using librosa.

    Parameters
    ----------
    y_percussive: np.array
        Percussive part of the audio signal in samples.
    sr: int
        Sample rate.

    Returns
    -------
    beats_idx: np.array
        Indeces in frames of the estimated beats.
    beats_times: np.array
        Time of the estimated beats.
    """
    logging.info("Estimating Beats...")
    tempo, beats_idx = librosa.beat.beat_track(y=y_percussive, sr=sr,
                                               hop_length=HOP_SIZE)
    return beats_idx, librosa.frames_to_time(beats_idx, sr=sr,
                                             hop_length=HOP_SIZE)


def compute_features(audio_file, type=PCP_TYPE):
    """
    Computes the audio features from a given audio file.

    Parameters
    ----------
    audio_file : str
        Path to the input audio file.
    type : str
        Feature type identification (must be one of the FEATURE_TYPES).

    Returns
    -------
    features : dict
        The desired features, including beat-synchronous features.
    """
    # Sanity check
    assert type in FEATURE_TYPES, "Incorrect features"

    # Load Audio
    logging.info("Loading audio file %s" % os.path.basename(audio_file))
    audio, sr = librosa.load(audio_file, sr=SAMPLING_RATE)

    # Compute harmonic-percussive source separation
    logging.info("Computing Harmonic Percussive source separation...")
    y_harmonic, y_percussive = librosa.effects.hpss(audio)

    # Output features dict
    features = {}

    if type == PCP_TYPE:
        logging.info("Computing Pitch Class Profiles...")
        features["sequence"] = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, hop_length=HOP_SIZE).T

    elif type == TONNETZ_TYPE:
        logging.info("Computing Tonnetz...")
        features["sequence"] = librosa.feature.tonnetz(y=y_harmonic, sr=sr).T

    elif type == MFCC_TYPE:
        logging.info("Computing Spectrogram...")
        S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=FFT_SIZE,
                                           hop_length=HOP_SIZE,
                                           n_mels=N_MELS)

        logging.info("Computing MFCCs...")
        log_S = librosa.logamplitude(S, ref_power=np.max)
        features["sequence"] = librosa.feature.mfcc(S=log_S, n_mfcc=N_MFCCS).T

    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()

    # Estimate Beats
    features["beats_idx"], features["beats"] = compute_beats(y_percussive,
                                                             sr=sr)

    # Compute Beat-sync features
    features["bs_sequence"] = librosa.feature.sync(features["sequence"].T,
                                                   features["beats_idx"],
                                                   pad=False).T
    return features


def compute_compression_measure(sequence, summary):
    """Computes the compression measure given.

    Parameters
    ----------
    sequence : np.array((M, n_features))
        Sequence of features.
    summary : list
        List of P np.arrays of shape(N, n_features)

    Returns
    -------
    compression : float >= 0
        The compression measure
    """
    # If the summary is empty, the compression should be empty
    if len(summary) == 0:
        return 0

    # Make sure that the dimensions are correct
    assert sequence.shape[1] == summary[0].shape[1]

    for subsequence in summary:
        for i in xrange(sequence.shape[0]):
            #TODO
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        "Generates a music summary from a given audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_file",
                        action="store",
                        help="Input audio file")
    parser.add_argument("-o",
                        action="store",
                        dest="out_file",
                        type=str,
                        help="Output file",
                        default="summary.wav")
    parser.add_argument("-s",
                        action="store_true",
                        dest="sonify_beats",
                        help="Sonifies the estimated beats",
                        default=False)
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (only for collection mode)",
                        default=4)
    parser.add_argument("-ow",
                        action="store_true",
                        dest="overwrite",
                        help="Overwrite the previously computed features",
                        default=False)
    parser.add_argument("-d",
                        action="store",
                        dest="ds_name",
                        default="*",
                        help="The prefix of the dataset to use "
                        "(e.g. Isophonics, SALAMI)")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    #TODO

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


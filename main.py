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
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import six
import time

from scipy.io import wavfile
from scipy.spatial import distance

import librosa
from utils import Segment
import utils

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


def compute_beats(y_percussive, sr):
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
    audio : np.array
        The samples of the audio
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

    # Normalize sequence
    features["sequence"] = utils.normalize(features["sequence"])

    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()

    # Estimate Beats
    features["beats_idx"], features["beats"] = compute_beats(y_percussive,
                                                             sr=sr)

    # Compute Beat-sync features
    features["bs_sequence"] = librosa.feature.sync(features["sequence"].T,
                                                   features["beats_idx"],
                                                   pad=False).T
    return features, audio


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

    # Get the fixed parameters
    P = len(summary)
    F = summary[0].shape[1]
    N = summary[0].shape[0]
    M = sequence.shape[0]
    J = M - N + 1

    # Make sure that the length of each subsequence of the summary is less
    # than the whole track
    assert M > N

    # Compute the "convolutive" euclidean distance
    dist = 0
    for gamma in summary:
        for i in np.arange(J):
            subsequence = sequence[i:i + N, :]
            X = np.vstack((gamma.flatten(), subsequence.flatten()))
            dist += distance.pdist(X, metric="sqeuclidean")[0] / float(N * F)

    # Normalize and transform to similarity
    return 1 - dist / float(P * J)


def compute_disjoint_information(summary, L):
    """Computes the disjoint information measure.

    Parameters
    ----------
    summary : list
        List of P np.arrays of shape (N, n_features)
    L : int > 0 < N
        The length of each shingle.

    Returns
    -------
    disjoint : float >= 0
        The disjoint information measure
    """

    # Sanity checks
    assert len(summary) > 0
    P = len(summary)
    N = len(summary[0])
    assert L < N
    for subsequence in summary:
        assert N == len(subsequence)

    # If there's only one subsequence in the summary, return maximum measure
    if P == 1:
        return 1.0

    # Compute the measure
    disjoint = 1
    for i in np.arange(P):
        for j in np.arange(i + 1, P):
            shingles_i = make_shingles(summary[i], L)
            shingles_j = make_shingles(summary[j], L)
            disjoint *= compute_avg_min_dist(shingles_i, shingles_j)

    # Normalize
    disjoint **= (2.0 / float(P * (P - 1)))

    return disjoint


def compute_summary_criterion(sequence, summary, L):
    """Computes the music summary criterion.

    Parameters
    ----------
    sequence : np.array((M, n_features))
        Sequence of features.
    summary : list
        List of P np.arrays of shape(N, n_features)
    L : int > 0 < N
        The length of each shingle.

    Returns
    -------
    criterion : float >= 0
        The music summary criterion
    compression : float >= 0
        The compression measure
    disjoint : float >= 0
        The disjoint information measure
    """
    compression = compute_compression_measure(sequence, summary)
    disjoint = compute_disjoint_information(summary, L)
    criterion = utils.f_measure(compression, disjoint)
    return criterion, compression, disjoint


def make_shingles(subsequence, L):
    """Transforms the given subsequence into a list of L-length shingles.

    Parameters
    ----------
    subsequence : np.array(N, n_features)
        A given subsequence of the track.
    L : int > 0 < N
        The length of each shingle.

    Returns
    -------
    shingles : list
        List of np.array(L, n_features) representing the shingles.
    """
    # Get sizes
    N = subsequence.shape[0]
    K = N - L + 1

    # Some sanity checks
    assert N > 0
    assert N >= L

    # Make shingles
    shingles = []
    for i in np.arange(K):
        shingles.append(subsequence[i:i + L, :])

    return shingles


def compute_avg_min_dist(shingles1, shingles2):
    """Computes the averaged minimum Euclidean distance between two sets of
    shingles.

    Parameters
    ----------
    shingles1: list
        List of np.arrays for a given subsequence.
    shingles2: list
        List of np.arrays for a given subsequence.

    Returns
    -------
    min_dist : float > 0
        Averaged minimum Euclidean distance between the two sets of shingles.
    """
    # Some sanity checks
    assert len(shingles1) == len(shingles2)

    # If shingle is empty, no distance exists
    if len(shingles1) <= 0:
        return None

    # Get fixed parameters
    L = shingles1[0].shape[0]
    K = len(shingles1)

    # Find average minimum distance
    avg_min_dist = 0
    for shingle1 in shingles1:
        min_dist = np.inf
        for shingle2 in shingles2:
            X = np.vstack((shingle1.flatten(), shingle2.flatten()))
            dist = distance.pdist(X, metric="sqeuclidean")[0] / float(L)
            if dist < min_dist:
                min_dist = dist
        avg_min_dist += min_dist

    # Normalize
    return avg_min_dist / float(K)


def find_optimal_summary(sequence, P, N, L=None):
    """Identifies the optimal summary of the sequence.

    Parameters
    ----------
    sequence : np.array(M, n_features)
        Representation of the audio track.
    P : int > 0
        Number of subsequences in the summary.
    N : int > 0
        Numnber of beats per subsequence.
    L : int > 0 < N
        Length of the shingles (If None, L = N / 2)

    Returns
    -------
    summary_idxs : list (len == P)
        List of indeces of starting points of the summary.
    """
    # Sanity checks
    assert len(sequence) >= P * N

    M = len(sequence)

    # Create tensors to store the two measures
    disjoints = np.zeros(P * (M, ))
    compressions = np.zeros(P * (M, ))
    criteria = np.zeros(P * (M, ))

    # Get all the possible subsequences (shingles)
    subsequences = make_shingles(sequence, N)
    n = len(subsequences)

    # Get all possible P combinations
    combs = itertools.combinations(subsequences, P)
    combs_idxs = itertools.combinations(np.arange(n), P)

    # Compute measures
    for comb, idx in zip(combs, combs_idxs):
        summary = list(comb)
        compressions[idx], disjoints[idx], criteria[idx] = \
            compute_summary_criterion(sequence, summary, L)

    summary_idxs = np.unravel_index(criteria.argmax(), criteria.shape)
    #six.print_(summary_idxs)
    #plt.imshow(criteria, interpolation="nearest")
    #plt.show()
    #plt.imshow(compressions, interpolation="nearest")
    #plt.show()
    #plt.imshow(disjoints, interpolation="nearest")
    #plt.show()

    return summary_idxs


def find_heur_summary(sequence, P, N, L=None):
    """Identifies the summary of the given sequence using the heuristic
    approach.

    Parameters
    ----------
    sequence : np.array(M, n_features)
        Representation of the audio track.
    P : int > 0
        Number of subsequences in the summary.
    N : int > 0
        Numnber of beats per subsequence.
    L : int > 0 < N
        Length of the shingles (If None, L = N / 2)

    Returns
    -------
    summary_idxs : list (len == P)
        List of indeces of starting points of the summary.
    """
    # Sanity checks
    assert len(sequence) >= P * N

    M = len(sequence)
    if L is None:
        L = int(N / 2)

    # Initial positions of the subsequences, uniformly spread
    summary_idxs = np.round(np.arange(0, M,
                                      M / float(P) / float(2.0))[1::2] - N)
    summary_idxs = np.asarray(summary_idxs, dtype=int)

    # Make sure that our summary has the right amount of subsequences
    assert len(summary_idxs) == P

    # For each subsequence, move it until finding the max
    for i, start_idx in enumerate(summary_idxs):
        # Set start and end points
        if i == 0:
            start_idx = 0
        else:
            start_idx = summary_idxs[i - 1] + N
        if i == P - 1:
            end_idx = M - N
        else:
            end_idx = summary_idxs[i + 1] - N

        # Find maximum criterion for current index
        max_criterion = 0
        max_idx = 0
        for curr_idx in np.arange(start_idx, end_idx):
            summary_idxs[i] = curr_idx
            # Construct summary correctly
            summary = []
            for idx in summary_idxs:
                summary.append(sequence[idx:idx + N])

            # Compute criterion
            criterion, compression, disjoint = \
                compute_summary_criterion(sequence, summary, L)
            if criterion > max_criterion:
                max_criterion = criterion
                max_idx = curr_idx

        # Update summary indeces
        summary_idxs[i] = max_idx

    return summary_idxs


def synth_summary(audio, beats, summary_idxs, N, fade=2):
    """Synthesize the audio summary.

    Parameters
    ----------
    audio : np.array
        Samples of the audio, sampled at SAMPLING_RATE.
    beats : np.array
        Beat times of the audio.
    summary_idxs : list
        List of the starting beat indeces of the summary.
    N : int
        Number of beats of each subsequence of the summary.
    fade : int
        Number of beats of cross-fading.

    Returns
    -------
    summary : np.array
        Samples of the summary, sampled at SAMPLING_RATE.
    """
    # Sanity checks
    assert fade % 2 == 0

    n = len(audio)

    summary = np.empty(0)
    for i, idx in enumerate(summary_idxs):
        # Create Subsequence
        subseq = Segment(beats[idx + fade / 2], beats[idx + N - fade / 2],
                         sr=SAMPLING_RATE)
        subseq_audio = audio[subseq.start_sample:subseq.end_sample]

        # Create cross fade segments
        fade_in_seg = Segment(beats[np.max([0, idx - fade / 2])],
                              beats[np.min([n - 1, idx + fade / 2])],
                              sr=SAMPLING_RATE)
        fade_out_seg = Segment(beats[np.max([0, idx + N - fade / 2])],
                              beats[np.min([n - 1, idx + N + fade / 2])],
                              sr=SAMPLING_RATE)

        # Create fade in/out audio
        fade_in = utils.generate_fade_audio(fade_in_seg, audio, is_out=False)
        fade_out = utils.generate_fade_audio(fade_out_seg, audio, is_out=True)

        if i == 0:
            # Beginning of track
            summary = np.concatenate((fade_in, subseq_audio))
        else:
            # Middle of track
            if fade_in_seg.n_samples > 0:
                summary[-fade_in_seg.n_samples:] += fade_in
            summary = np.concatenate((summary, subseq_audio))

        # End of subsequence
        summary = np.concatenate((summary, fade_out))

    return summary


def generate_summary(audio_file, P=3, N=16, L=None, feature_type=PCP_TYPE,
                     opt=False, out_file=None):
    """Generates the summary of a given audio file.

    Parameters
    ----------
    audio_file : str
        Path to the input audio file.
    P : int > 0
        Number of subsequences in the summary.
    N : int > 0
        Numnber of beats per subsequence.
    L : int > 0 < N
        Length of the shingles (If None, L = N / 2)
    opt : bool
        Whether to use the optimal or the heuristic method.
    out_file : str
        Path to the output summary audio file (None to not save output).

    Returns
    -------
    summary : np.array
        Samples of the final summary.
    """
    # Compute audio features
    features, audio = compute_features(audio_file, type=feature_type)

    # Find summary
    if opt:
        summary_idxs = find_optimal_summary(features["bs_sequence"], P, N, L)
    else:
        summary_idxs = find_heur_summary(features["bs_sequence"], P, N, L)

    # Synthesize summary
    summary = synth_summary(audio, features["beats"], summary_idxs, N)

    # Save file if needed
    if out_file is not None:
        wavfile.write(out_file, SAMPLING_RATE, summary)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        "Generates a music summary from a given audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_file",
                        action="store",
                        help="Input audio file")
    parser.add_argument("-P",
                        action="store",
                        dest="P",
                        type=int,
                        help="Number of subsequences in summary",
                        default=3)
    parser.add_argument("-N",
                        action="store",
                        dest="N",
                        type=int,
                        help="Number of beats per subsequence",
                        default=16)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    #TODO

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

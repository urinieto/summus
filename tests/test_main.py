#!/usr/bin/env python
"""Unit Tests for the Music Summaries code."""

import numpy as np
import os
import six
import sys

from nose.tools import eq_, raises

sys.path.append("..")
import main

AUDIO_DIR = os.path.join("..", "audio")


def test_avg_min_dist():
    def _test_avg_min_dist(sub1, sub2, L, result):
        shingles1 = main.make_shingles(sub1, L)
        shingles2 = main.make_shingles(sub2, L)
        avg_min_dist = main.compute_avg_min_dist(shingles1, shingles2)
        assert np.isclose(avg_min_dist, result)

    L = 3
    N = 10
    sub1 = np.arange(N).reshape(N, 1)
    sub2 = np.arange(N).reshape(N, 1) + 1
    _test_avg_min_dist(sub1, sub2, L, 1 / 8.)
    L = 4
    N = 10
    sub1 = np.arange(N).reshape(N, 1)
    sub2 = np.arange(N).reshape(N, 1) + 100
    _test_avg_min_dist(sub1, sub2, L, 36.6703)


def test_make_shingles():
    def _test_shingles(N, n_features, L):
        K = N - L + 1
        subsequence = np.arange(N * n_features).reshape(N, n_features)
        shingles = main.make_shingles(subsequence, L)
        eq_(len(shingles), K, "The number of shingles is not correct")
        eq_(shingles[0].shape[0], L, "The size of the shingles is not correct")
        eq_(shingles[0].shape[1], n_features, "The number of features of the "
            "shingles is not correct")
    N = 8
    n_features = 2
    L = 3
    _test_shingles(N, n_features, L)
    N = 10
    n_features = 4
    L = 4
    _test_shingles(N, n_features, L)
    N = 1
    n_features = 4
    L = 1
    _test_shingles(N, n_features, L)


def test_compression_measure():
    # Check that a toy example actually returns perfect compression
    sequence = np.ones(20)[:, np.newaxis]
    summary = [np.ones((3, 1))]
    compression = main.compute_compression_measure(sequence, summary)
    assert np.isclose(compression, 1.0)

    # Check that a toy example actually returns zero compression
    sequence = np.zeros(20)[:, np.newaxis]
    summary = [np.ones((3, 1))]
    compression = main.compute_compression_measure(sequence, summary)
    assert np.isclose(compression, 0.0)


def test_compute_features():
    audio_file = os.path.join(AUDIO_DIR, "sines.ogg")

    # Chromagram
    chroma = main.compute_features(audio_file, main.PCP_TYPE)
    eq_(chroma["sequence"].shape[1], 12, "Chromagram is not 12-dimensional")

    # Tonnetz
    tonnetz = main.compute_features(audio_file, main.TONNETZ_TYPE)
    eq_(tonnetz["sequence"].shape[1], 6, "Tonnetz is not 6-dimensional")

    # MFCC
    mfcc = main.compute_features(audio_file, main.MFCC_TYPE)
    eq_(mfcc["sequence"].shape[1], main.N_MFCCS, "MFCC have not the right "
        "number of coefficients")

    # Check that all features have the same length
    assert chroma["sequence"].shape[0] == tonnetz["sequence"].shape[0] and \
        tonnetz["sequence"].shape[0] == mfcc["sequence"].shape[0]


@raises(AssertionError)
def test_compute_wrong_features():
    audio_file = os.path.join(AUDIO_DIR, "sines.ogg")
    features = main.compute_features(audio_file, type="fail")
    if features:
        return  # This should not get executed

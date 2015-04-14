#!/usr/bin/env python
"""Unit Tests for the Music Summaries code."""

import numpy as np
import os
import sys

from nose.tools import eq_, raises

sys.path.append("..")
import main

AUDIO_DIR = os.path.join("..", "audio")


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



#!/usr/bin/env python
"""Unit Tests for the Music Summaries code."""

import os
import sys

sys.path.append("..")
import main

AUDIO_DIR = os.path.join("..", "audio")


def test_compute_features():
    audio_file = os.path.join(AUDIO_DIR, "sines.ogg")

    # Chromagram
    chroma = main.compute_features(audio_file, main.PCP_TYPE)
    assert chroma["sequence"].shape[1] == 12

    # Tonnetz
    tonnetz = main.compute_features(audio_file, main.TONNETZ_TYPE)
    assert tonnetz["sequence"].shape[1] == 6

    # MFCC
    mfcc = main.compute_features(audio_file, main.MFCC_TYPE)
    assert mfcc["sequence"].shape[1] == main.N_MFCCS

    # Check that all the features have the same length
    assert chroma["sequence"].shape[0] == tonnetz["sequence"].shape[0] and \
        tonnetz["sequence"].shape[0] == mfcc["sequence"].shape[0]

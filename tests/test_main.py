#!/usr/bin/env python
"""Unit Tests for the Music Summaries code."""

import os
import sys

sys.path.append("..")
import main

AUDIO_DIR = os.path.join("..", "audio")


def test_compute_features():
    audio_file = os.path.join(AUDIO_DIR, "sines.ogg")
    features = main.compute_features(audio_file, main.PCP_TYPE)
    print features["sequence"].shape

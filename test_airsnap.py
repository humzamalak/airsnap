"""Tests for Airsnap: synthetic Doppler frames through the real detector.

Run: ~/.hermes/hermes-agent/venv/bin/python test_airsnap.py
No audio hardware needed.
"""
import numpy as np

import airsnap

FP = airsnap.PILOT_BIN * airsnap.BIN_HZ
T = np.arange(airsnap.FFT_SIZE) / airsnap.SAMPLE_RATE
RNG = np.random.default_rng(7)


def synth_frame(lo=0, hi=0, amp=0.25):
    """Raw mic frame: pilot tone + band-limited reflection [lo,hi] bins off."""
    frame = 0.5 * np.sin(2 * np.pi * FP * T)
    if lo or hi:
        for f in np.linspace(FP + lo * airsnap.BIN_HZ,
                             FP + hi * airsnap.BIN_HZ, 12):
            frame += amp * np.sin(2 * np.pi * f * T + RNG.uniform(0, 2 * np.pi))
    return frame.astype(np.float32)


def test_pilot_bin():
    assert airsnap.PILOT_BIN == 810
    assert abs(airsnap.BIN_HZ - 23.4375) < 0.01


def test_silence():
    d = airsnap.DopplerDetector()
    assert d.analyze(np.zeros(airsnap.FFT_SIZE)) == (0, 0.0)


def test_no_motion():
    d = airsnap.DopplerDetector()
    direction, _ = d.analyze(synth_frame())
    assert direction == 0, "pure pilot + noise reflection should not fire"


def test_toward():
    d = airsnap.DopplerDetector()
    direction, strength = d.analyze(synth_frame(4, 10))
    assert direction == 1, f"toward should be +1, got {direction} ({strength})"
    assert strength > airsnap.ASYM_THRESHOLD


def test_away():
    d = airsnap.DopplerDetector()
    direction, strength = d.analyze(synth_frame(-10, -4))
    assert direction == -1, f"away should be -1, got {direction} ({strength})"


def test_event_ratelimit():
    d = airsnap.DopplerDetector()
    assert d.event(0.5, now=1.0) is True
    assert d.event(0.5, now=1.03) is False
    assert d.event(0.5, now=1.20) is True


if __name__ == "__main__":
    for fn in [test_pilot_bin, test_silence, test_no_motion,
               test_toward, test_away, test_event_ratelimit]:
        fn()
        print(f"PASSED {fn.__name__}")

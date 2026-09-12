"""Airsnap: camera-free scroll control via ultrasonic Doppler sensing.

Plays an inaudible 19 kHz pilot tone through the speakers; hand motion near
the laptop Doppler-shifts the reflection picked up by the microphone. FFT
sideband-energy analysis turns the shift into directional motion events,
which scroll the page up/down.

Technique based on SoundWave (Gupta, Morris, Patel, Tan — CHI 2012).

Usage:
    python airsnap.py                # readout mode (no scrolling)
    python airsnap.py --scroll      # actually scroll
    python airsnap.py --freq 18500  # lower pilot if speakers roll off
    python airsnap.py --list-devices
"""

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

# ---------- Tuning ----------
PILOT_FREQ = 19000
SAMPLE_RATE = 48000
FFT_SIZE = 2048
BIN_HZ = SAMPLE_RATE / FFT_SIZE            # 23.4 Hz per bin
PILOT_BIN = int(PILOT_FREQ / BIN_HZ)       # 810
PILOT_SKIRT = 3                            # bins either side of pilot to skip
SIDEBAND_SPAN = 12                         # bins of sideband to integrate
SIDEBAND_RATIO = 0.15                      # (legacy) sideband/peak ratio
ASYM_THRESHOLD = 0.5                       # |asymmetry| to fire (rest ≈ 0.2)
SNR_MIN = 20.0                             # pilot must exceed noise floor
MIN_EVENT_INTERVAL = 0.06                  # s between motion events
SCROLL_SCALE = 2.0                         # scroll lines per unit strength

window = np.hamming(FFT_SIZE)


class DopplerDetector:
    """Sideband-energy detector: motion shifts reflected energy above
    (toward) or below (away) the pilot peak. Energy in the sidebands vs the
    pilot peak gives both direction and a velocity proxy."""

    def __init__(self, pilot_bin=PILOT_BIN):
        self.pilot_bin = pilot_bin
        self.last_event = 0.0

    def analyze(self, frame):
        """Return (direction, strength) for a raw mic frame.

        direction: -1 toward (scroll up), +1 away (scroll down), 0 none.
        strength: sideband asymmetry in [-1, 1] (velocity proxy).

        Uses up-vs-down sideband asymmetry rather than absolute sideband
        energy — speaker distortion and room reflections give a roughly
        symmetric baseline, so asymmetry cancels it out.
        """
        spec = np.abs(np.fft.rfft(frame * window))
        lo, hi = self.pilot_bin - 3, self.pilot_bin + 4
        k = lo + int(np.argmax(spec[lo:hi]))
        peak = spec[k]
        noise = np.median(spec[100:700])
        if peak < max(noise * SNR_MIN, 1e-9):
            return 0, 0.0

        p = self.pilot_bin
        up = float(np.sum(spec[p + PILOT_SKIRT:p + PILOT_SKIRT + SIDEBAND_SPAN]))
        down = float(np.sum(spec[p - PILOT_SKIRT - SIDEBAND_SPAN:p - PILOT_SKIRT]))
        asym = (up - down) / (up + down + 1e-9)
        if abs(asym) < ASYM_THRESHOLD:
            return 0, 0.0
        return (1, asym) if asym > 0 else (-1, -asym)

    def event(self, strength, now=None):
        """Rate-limit events. True if this one should fire."""
        now = now or time.time()
        if now - self.last_event < MIN_EVENT_INTERVAL:
            return False
        self.last_event = now
        return True


def scroll_native(direction, strength):
    """CGEvent scroll — smooth, no third-party GUI deps."""
    import Quartz
    lines = int(max(1, strength * 40 * SCROLL_SCALE))
    # direction -1 = hand toward laptop = scroll up (wheel up)
    delta = -lines if direction < 0 else lines
    ev = Quartz.CGEventCreateScrollWheelEvent(
        None, Quartz.kCGScrollEventUnitLine, 1, delta)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


class Airsnap:
    def __init__(self, freq=PILOT_FREQ, gain=0.3, scroll=False, verbose=True):
        self.gain = gain
        self.scroll = scroll
        self.verbose = verbose
        self.detector = DopplerDetector(int(freq / BIN_HZ))
        phase = np.arange(FFT_SIZE) * 2 * np.pi * freq / SAMPLE_RATE
        self.tone = (gain * np.sin(phase)).astype(np.float32)
        self.tone_pos = 0

    def _duplex(self, indata, outdata, frames, t, status):
        if status:
            print(f"[stream: {status}]", file=sys.stderr)
        # output: continuous pilot tone
        n = len(outdata)
        idx = self.tone_pos % len(self.tone)
        chunk = self.tone[idx:idx + n]
        if len(chunk) < n:
            chunk = np.concatenate([chunk, self.tone[:n - len(chunk)]])
        outdata[:] = chunk.reshape(-1, 1)
        self.tone_pos += n
        # input: detect doppler motion
        direction, strength = self.detector.analyze(indata[:, 0])
        if direction == 0 or not self.detector.event(strength):
            return
        if self.scroll:
            scroll_native(direction, strength)
        if self.verbose:
            label = "TOWARD scroll-up  " if direction < 0 else "AWAY   scroll-down"
            print(f"{label} strength={strength:.2f}")

    def run(self):
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE, blocksize=FFT_SIZE, channels=1,
            dtype="float32", callback=self._duplex)
        mode = "scrolling" if self.scroll else "readout"
        print(f"Airsnap: pilot {PILOT_FREQ / 1000:.1f} kHz, mode={mode}")
        print("Move your hand toward/away from the laptop. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.stream.stop()
            self.stream.close()


def main():
    ap = argparse.ArgumentParser(description="Airsnap doppler scroll")
    ap.add_argument("--freq", type=float, default=PILOT_FREQ)
    ap.add_argument("--gain", type=float, default=0.3)
    ap.add_argument("--scroll", action="store_true", help="post scroll events")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    Airsnap(freq=args.freq, gain=args.gain, scroll=args.scroll).run()


if __name__ == "__main__":
    main()

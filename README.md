# Airsnap

Camera-free gesture scrolling for macOS using ultrasonic Doppler sensing —
no camera, no extra hardware. The laptop's own speakers emit an inaudible
~19 kHz pilot tone; a hand moving near the microphone Doppler-shifts the
reflection; an FFT pipeline converts the shift into scroll up/down events.

Based on the SoundWave technique (Gupta, Morris, Patel, Tan — Microsoft
Research / UW UbiComp Lab, CHI 2012).

## Status

**Working (verified live on this MacBook):** pilot tone playback + mic capture
duplex stream, 19 kHz pilot detected with SNR > 1000, sideband-asymmetry
detector (silence at rest: 0 false events in 3 s), CGEvent scroll posting.
Phase 1–2 committed. Remaining: real-hand tuning session (thresholds
`ASYM_THRESHOLD`, `MIN_EVENT_INTERVAL`, `SCROLL_SCALE`), menu-bar app,
optional calibration sweep.

## Requirements

- macOS with built-in speakers + microphone
- Python 3.9+
- `pip install numpy sounddevice`
- Microphone permission for your terminal app (first run prompts)

## Usage

```bash
python airsnap.py                 # 19 kHz pilot, live motion readout
python airsnap.py --freq 18500    # if speakers roll off too high
python airsnap.py --list-devices  # inspect audio devices
```

Move your hand toward the laptop → `TOWARD (scroll up)`;
away → `AWAY (scroll down)`. Strength is the Doppler bandwidth in bins
(velocity proxy).

## Science in one paragraph

A stationary speaker emits a continuous tone at ~19 kHz (inaudible to most
adults). Sound reflects off your hand; motion shifts the reflected frequency
by the Doppler formula Δf ≈ 2·f·v/c (v = hand speed, c = speed of sound —
a 0.4 m/s hand at 19 kHz shifts ~44 Hz). The microphone is sampled at
48 kHz; 2048-point Hamming-windowed FFTs (~21.5 Hz/bin) reveal the pilot
peak. Bandwidth spreading ≥ 4 bins around the peak is a motion event; the
side of the peak (higher/lower frequency) gives direction — toward/away.

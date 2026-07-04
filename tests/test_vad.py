import math
import struct

from app.realtime.vad import SPEECH_END, SPEECH_START, VadStream, frame_rms

RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000


def tone(ms: int, amplitude: int = 8000, freq: float = 220.0) -> bytes:
    n = RATE * ms // 1000
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


def silence(ms: int) -> bytes:
    n = RATE * ms // 1000
    return b"\x00\x00" * n


def make_vad(**overrides) -> VadStream:
    params = dict(
        sample_rate=RATE,
        frame_ms=FRAME_MS,
        start_ms=100,
        end_silence_ms=300,
        min_utterance_ms=200,
        pre_roll_ms=100,
        min_rms=0.010,
        noise_multiplier=3.0,
        barge_start_ms=300,
        barge_noise_multiplier=5.0,
    )
    params.update(overrides)
    return VadStream(**params)


def collect(vad: VadStream, pcm: bytes):
    events = []
    # Feed in odd-sized chunks to exercise reframing.
    step = 700
    for i in range(0, len(pcm), step):
        events.extend(vad.feed(pcm[i : i + step]))
    return events


def test_frame_rms_scales():
    assert frame_rms(silence(20)) == 0.0
    loud = frame_rms(tone(20, amplitude=16000))
    quiet = frame_rms(tone(20, amplitude=1600))
    assert loud > quiet > 0.0


def test_detects_utterance_with_endpoint():
    vad = make_vad()
    events = collect(vad, silence(400) + tone(600) + silence(500))
    kinds = [k for k, _ in events]
    assert kinds == [SPEECH_START, SPEECH_END]
    utterance = events[1][1]
    # Utterance should contain at least the voiced 600ms.
    assert len(utterance) >= len(tone(600))


def test_short_blip_is_ignored():
    vad = make_vad()
    events = collect(vad, silence(400) + tone(60) + silence(600))
    assert events == []  # never even reached speech-start sustain


def test_two_utterances_two_endpoints():
    vad = make_vad()
    pcm = silence(300) + tone(500) + silence(400) + tone(500) + silence(400)
    kinds = [k for k, _ in collect(vad, pcm)]
    assert kinds == [SPEECH_START, SPEECH_END, SPEECH_START, SPEECH_END]


def test_barge_mode_requires_longer_sustain():
    # In barge mode a 160ms burst (>= normal 100ms start) must NOT trigger.
    vad = make_vad()
    vad.set_barge_mode(True)
    events = collect(vad, silence(300) + tone(160) + silence(400))
    assert events == []

    # But sustained talking (>= 300ms) does trigger.
    vad2 = make_vad()
    vad2.set_barge_mode(True)
    events2 = collect(vad2, silence(300) + tone(600) + silence(400))
    kinds2 = [k for k, _ in events2]
    assert SPEECH_START in kinds2


def test_barge_mode_reset_clears_mid_speech_state():
    vad = make_vad()
    collect(vad, silence(300) + tone(200))  # mid-speech now
    assert vad.in_speech
    vad.set_barge_mode(True)
    assert not vad.in_speech  # turn boundary wipes the half-built utterance

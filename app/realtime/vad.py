"""Energy-based voice activity detection with endpointing.

Pure Python, zero dependencies — good enough for "phone on a desk" and
easily swappable for Silero later (the session only consumes the events).

Feed raw PCM16 mono bytes in any chunk size; the detector reframes to
``frame_ms`` internally and emits:

* ``SPEECH_START`` — sustained voice detected (after ``start_ms``)
* ``SPEECH_END``   — the speaker paused for ``end_silence_ms``; carries the
  full utterance audio including a little pre-roll from before the start.

An adaptive noise floor (EMA over non-speech frames) keeps it stable across
quiet rooms and noisy fans. While Vyra is speaking, the session switches
the detector to *barge-in mode*: a higher threshold and a longer sustain
requirement, so the phone's own speaker doesn't count as an interruption.
"""

import math
import struct
from typing import List, Optional, Tuple

SPEECH_START = "speech_start"
SPEECH_END = "speech_end"

Event = Tuple[str, bytes]


class VadStream:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        start_ms: int = 100,
        end_silence_ms: int = 700,
        min_utterance_ms: int = 300,
        pre_roll_ms: int = 200,
        min_rms: float = 0.010,
        noise_multiplier: float = 3.0,
        barge_start_ms: int = 300,
        barge_noise_multiplier: float = 5.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self._frame_ms = frame_ms
        self._start_frames = max(1, start_ms // frame_ms)
        self._end_frames = max(1, end_silence_ms // frame_ms)
        self._min_utt_frames = max(1, min_utterance_ms // frame_ms)
        self._pre_roll_frames = max(0, pre_roll_ms // frame_ms)
        self._min_rms = min_rms
        self._mult = noise_multiplier
        self._barge_start_frames = max(1, barge_start_ms // frame_ms)
        self._barge_mult = barge_noise_multiplier

        self._buffer = b""
        self._noise = 0.006  # EMA noise floor, full-scale fraction
        self._barge_mode = False

        self._in_speech = False
        self._voiced_run = 0
        self._silence_run = 0
        self._recent: List[bytes] = []  # pre-roll ring
        self._utterance: List[bytes] = []

    # --- Mode -----------------------------------------------------------
    def set_barge_mode(self, enabled: bool) -> None:
        """Stricter detection while Vyra talks (avoids speaker echo)."""
        if enabled != self._barge_mode:
            self._barge_mode = enabled
            self._voiced_run = 0
            if enabled:
                # A turn boundary: whatever was mid-flight no longer applies.
                self._in_speech = False
                self._silence_run = 0
                self._utterance = []

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    # --- Feeding --------------------------------------------------------
    def feed(self, pcm: bytes) -> List[Event]:
        events: List[Event] = []
        self._buffer += pcm
        while len(self._buffer) >= self.frame_bytes:
            frame = self._buffer[: self.frame_bytes]
            self._buffer = self._buffer[self.frame_bytes :]
            events.extend(self._process_frame(frame))
        return events

    def _process_frame(self, frame: bytes) -> List[Event]:
        events: List[Event] = []
        rms = frame_rms(frame)
        start_frames = (
            self._barge_start_frames if self._barge_mode else self._start_frames
        )
        mult = self._barge_mult if self._barge_mode else self._mult
        threshold = max(self._min_rms, self._noise * mult)
        voiced = rms >= threshold

        if not self._in_speech:
            if voiced:
                self._voiced_run += 1
            else:
                self._voiced_run = 0
                # Only quiet frames teach the noise floor.
                self._noise = 0.95 * self._noise + 0.05 * rms
            self._push_recent(frame)
            if self._voiced_run >= start_frames:
                self._in_speech = True
                self._silence_run = 0
                self._utterance = list(self._recent)  # includes pre-roll
                events.append((SPEECH_START, b""))
        else:
            self._utterance.append(frame)
            if voiced:
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._silence_run >= self._end_frames:
                    utterance = b"".join(self._utterance)
                    voiced_frames = len(self._utterance) - self._silence_run
                    self._reset_speech()
                    if voiced_frames >= self._min_utt_frames:
                        events.append((SPEECH_END, utterance))
        return events

    def _push_recent(self, frame: bytes) -> None:
        if self._pre_roll_frames == 0:
            return
        self._recent.append(frame)
        if len(self._recent) > self._pre_roll_frames:
            self._recent.pop(0)

    def _reset_speech(self) -> None:
        self._in_speech = False
        self._voiced_run = 0
        self._silence_run = 0
        self._utterance = []
        self._recent = []


def frame_rms(frame: bytes) -> float:
    """RMS of a PCM16 frame as a fraction of full scale (0..1)."""
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame[: count * 2])
    acc = 0
    for sample in samples:
        acc += sample * sample
    return math.sqrt(acc / count) / 32768.0

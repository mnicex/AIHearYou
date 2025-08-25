#!/usr/bin/env python3
# Simple POC for recording audio from BT device, detect pauses, call transcript and display results.
#
# 1st version 23.8. mikan (with help Github Copilot!)
# 2nd version 25.6. mikan (refactored, added Azure STT) - First tests OK!!!!
# Requires: pip install sounddevice webrtcvad azure-cognitiveservices-speech numpy
# Tested with Python 3.8+
# 

import argparse
import os
import sys
import time
import wave
import tempfile
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from queue import Queue, Full, Empty
from typing import Generator, Optional, Tuple

import numpy as np
import sounddevice as sd
import webrtcvad
import azure.cognitiveservices.speech as speechsdk


@dataclass
class AudioConfig:
    samplerate: int = 16000          # webrtcvad supports 8000/16000/32000/48000
    channels: int = 1                # mono
    frame_ms: int = 30               # 10, 20, or 30 ms for webrtcvad
    dtype: str = "int16"             # 16-bit PCM for webrtcvad


@dataclass
class VadConfig:
    aggressiveness: int = 2          # 0-3 (3 is most aggressive)
    pre_roll_ms: int = 300           # keep some audio before first speech
    silence_end_ms: int = 600        # required trailing silence to end a segment
    min_segment_ms: int = 800        # discard shorter segments
    max_segment_ms: int = 30000      # safety cutoff


def list_devices() -> None:
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    for idx, d in enumerate(devices):
        is_default = " [default]" if idx == default_in else ""
        print(f"{idx:3d}: {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']}){is_default}")


def find_device_index(name_substring: str) -> Optional[int]:
    name_substring = name_substring.lower()
    for idx, d in enumerate(sd.query_devices()):
        if name_substring in d["name"].lower() and d["max_input_channels"] > 0:
            return idx
    return None


def bytes_per_frame(cfg: AudioConfig) -> int:
    samples_per_frame = int(cfg.samplerate * cfg.frame_ms / 1000)
    return samples_per_frame * cfg.channels * np.dtype(cfg.dtype).itemsize


def write_wav(path: str, pcm_bytes: bytes, cfg: AudioConfig) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(cfg.channels)
        wf.setsampwidth(np.dtype(cfg.dtype).itemsize)
        wf.setframerate(cfg.samplerate)
        wf.writeframes(pcm_bytes)


def segmenter(
    frames: Generator[bytes, None, None],
    cfg: AudioConfig,
    vad_cfg: VadConfig
) -> Generator[Tuple[bytes, float], None, None]:
    vad = webrtcvad.Vad(vad_cfg.aggressiveness)
    frame_dur = cfg.frame_ms / 1000.0
    pre_roll_frames = int(vad_cfg.pre_roll_ms / cfg.frame_ms)
    silence_end_frames = int(vad_cfg.silence_end_ms / cfg.frame_ms)
    max_frames = int(vad_cfg.max_segment_ms / cfg.frame_ms)
    min_frames = int(vad_cfg.min_segment_ms / cfg.frame_ms)

    ring = deque(maxlen=pre_roll_frames)
    triggered = False
    voiced_frames = []
    silence_run = 0
    total_frames_in_segment = 0

    for f in frames:
        is_voiced = vad.is_speech(f, cfg.samplerate)
        if not triggered:
            ring.append((f, is_voiced))
            num_voiced = sum(1 for _, v in ring if v)
            # Trigger when >50% of pre-roll frames are voiced (or immediate voiced if no pre-roll)
            if (pre_roll_frames > 0 and num_voiced > len(ring) // 2) or (pre_roll_frames == 0 and is_voiced):
                triggered = True
                voiced_frames.extend([rf for rf, _ in ring])
                ring.clear()
                total_frames_in_segment = len(voiced_frames)
                silence_run = 0
        else:
            voiced_frames.append(f)
            total_frames_in_segment += 1
            if is_voiced:
                silence_run = 0
            else:
                silence_run += 1

            if total_frames_in_segment >= max_frames or silence_run >= silence_end_frames:
                # finalize
                if total_frames_in_segment >= min_frames:
                    seg_bytes = b"".join(voiced_frames)
                    seg_sec = total_frames_in_segment * frame_dur
                    yield seg_bytes, seg_sec
                # reset
                triggered = False
                voiced_frames.clear()
                silence_run = 0
                total_frames_in_segment = 0

    # flush if ending while in a segment
    if triggered and total_frames_in_segment >= min_frames:
        yield b"".join(voiced_frames), total_frames_in_segment * frame_dur


def audio_frames(
    cfg: AudioConfig,
    device_index: Optional[int],
    queue_max: int = 128
) -> Generator[bytes, None, None]:
    q: Queue[bytes] = Queue(maxsize=queue_max)
    bpf = bytes_per_frame(cfg)

    def callback(indata, frames, time_info, status):
        if status:
            # Non-fatal; print once per callback if something happens
            print(f"[audio] status: {status}", file=sys.stderr)
        try:
            # Ensure int16 mono
            arr = np.asarray(indata, dtype=cfg.dtype)
            if arr.ndim > 1 and arr.shape[1] > 1:
                # downmix to mono by averaging channels
                arr = np.mean(arr, axis=1).astype(cfg.dtype)
            q.put_nowait(arr.tobytes())
        except Full:
            # Drop if consumer is slow
            pass

    blocksize = int(cfg.samplerate * cfg.frame_ms / 1000)

    stream = sd.InputStream(
        samplerate=cfg.samplerate,
        channels=cfg.channels,
        dtype=cfg.dtype,
        callback=callback,
        device=device_index,
        blocksize=blocksize,
    )

    with stream:
        while True:
            try:
                chunk = q.get(timeout=1.0)
            except Empty:
                continue
            if len(chunk) != bpf:
                # pad/truncate to exact frame length
                if len(chunk) < bpf:
                    chunk = chunk + bytes(bpf - len(chunk))
                else:
                    chunk = chunk[:bpf]
            yield chunk


def transcribe_azure_wav_bytes(
    wav_pcm_bytes: bytes,
    cfg: AudioConfig,
    azure_key: str,
    azure_region: str,
    language: str
) -> Tuple[str, Optional[float]]:
    # Write to a temporary WAV file, then call recognize_once
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        write_wav(path, wav_pcm_bytes, cfg)
        speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
        speech_config.speech_recognition_language = language

        audio_config = speechsdk.audio.AudioConfig(filename=path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # Azure returns duration as 100-ns ticks (int). Convert to seconds.
            dur_sec: Optional[float] = None
            try:
                dur = result.duration
                if isinstance(dur, (int, float)):
                    dur_sec = float(dur) / 10_000_000.0
                elif hasattr(dur, "total_seconds"):
                    dur_sec = float(dur.total_seconds())  # just in case SDK changes
            except Exception:
                dur_sec = None
            return result.text, dur_sec
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "", None
        else:
            # Canceled or error
            details = ""
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                details = f" Canceled: {cancellation_details.reason}. ErrorDetails: {cancellation_details.error_details}"
            return f"[Azure STT error]{details}", None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Stream from Bluetooth mic, VAD segment, and transcribe via Azure Speech.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--device", type=str, default=None, help="Input device name substring (case-insensitive).")
    parser.add_argument("--device-index", type=int, default=None, help="Input device index (overrides --device).")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate. Must be 8000/16000/32000/48000 for webrtcvad.")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0,1,2,3], help="WebRTC VAD aggressiveness 0-3.")
    parser.add_argument("--frame-ms", type=int, default=30, choices=[10,20,30], help="Frame size in ms.")
    parser.add_argument("--silence-ms", type=int, default=600, help="Trailing silence to end a segment.")
    parser.add_argument("--pre-roll-ms", type=int, default=300, help="Audio to keep before first voiced frame.")
    parser.add_argument("--min-segment-ms", type=int, default=800, help="Discard segments shorter than this.")
    parser.add_argument("--max-segment-ms", type=int, default=30000, help="Safety cutoff per segment.")
    parser.add_argument("--language", type=str, default="en-US", help="Azure STT language, e.g., en-US, fi-FI, etc.")
    parser.add_argument("--save-segments", type=str, default=None, help="Optional folder to save segment WAVs for debugging.")
    parser.add_argument("--azure-key", type=str, default=None, help="Azure Speech key; falls back to AZURE_SPEECH_KEY env.")
    parser.add_argument("--azure-region", type=str, default=None, help="Azure Speech region; falls back to AZURE_SPEECH_REGION env.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Resolve device
    device_index = args.device_index
    if device_index is None and args.device:
        device_index = find_device_index(args.device)
        if device_index is None:
            print(f"Could not find input device matching: {args.device}", file=sys.stderr)
            sys.exit(2)

    # Azure creds
    azure_key = args.azure_key or os.getenv("AZURE_SPEECH_KEY")
    azure_region = args.azure_region or os.getenv("AZURE_SPEECH_REGION")
    if not azure_key or not azure_region:
        print("Azure credentials missing. Set --azure-key/--azure-region or AZURE_SPEECH_KEY/AZURE_SPEECH_REGION.", file=sys.stderr)
        sys.exit(3)

    a_cfg = AudioConfig(samplerate=args.sr, channels=1, frame_ms=args.frame_ms, dtype="int16")
    v_cfg = VadConfig(
        aggressiveness=args.vad_aggressiveness,
        pre_roll_ms=args.pre_roll_ms,
        silence_end_ms=args.silence_ms,
        min_segment_ms=args.min_segment_ms,
        max_segment_ms=args.max_segment_ms,
    )

    # Informational
    if device_index is None:
        di = sd.default.device[0]
        print(f"Using default input device index: {di}")
        device_index = di
    else:
        print(f"Using input device index: {device_index}")

    print("Starting capture. Press Ctrl+C to stop.")
    start_time = time.time()

    try:
        frames = audio_frames(a_cfg, device_index)
        seg_iter = segmenter(frames, a_cfg, v_cfg)
        seg_count = 0
        for seg_bytes, seg_sec in seg_iter:
            seg_count += 1
            rel_time = time.time() - start_time
            print(f"\n[Segment #{seg_count} | +{rel_time:0.2f}s | {seg_sec:0.2f}s] Transcribing...")

            # Optional save for debugging
            if args.save_segments:
                os.makedirs(args.save_segments, exist_ok=True)
                path = os.path.join(args.save_segments, f"segment_{seg_count:04d}.wav")
                write_wav(path, seg_bytes, a_cfg)
                print(f"Saved: {path}")

            text, _ = transcribe_azure_wav_bytes(
                wav_pcm_bytes=seg_bytes,
                cfg=a_cfg,
                azure_key=azure_key,
                azure_region=azure_region,
                language=args.language
            )

            if text.startswith("[Azure STT error]"):
                print(text)
            elif text.strip():
                print(f"Transcript: {text}")
            else:
                print("No speech recognized.")
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
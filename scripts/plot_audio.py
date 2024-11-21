#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "contourpy==1.2.1",
#     "cycler==0.12.1",
#     "fonttools==4.53.1",
#     "kiwisolver==1.4.5",
#     "matplotlib==3.9.2",
#     "numpy==2.1.0",
#     "packaging==24.1",
#     "pillow==10.4.0",
#     "pyparsing==3.1.2",
#     "python-dateutil==2.9.0.post0",
#     "six==1.16.0",
#     "Wave==0.0.2"
# ]
# ///
import wave
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio", "-a", help="audio file to plot")
parser.add_argument("--input", "-i", help="test data file to plot")
parser.add_argument("--output", "-o", help="file to save plot to (optional)")

args = parser.parse_args() 


wav_obj = wave.open(args.audio, 'rb')

speech_segments = []
silence_samples = []
speech_samples = []

def rust_duration_to_seconds(obj):
    return int(obj["secs"]) + (float(obj["nanos"]) / 1000000000.0)

redemption_time = 0

with open(args.input) as f:
    data = json.load(f)
    start = None
    end = None

    vad = data["summary"][args.audio]
    silence_samples = vad["current_silence_samples"]
    speech_samples = vad["current_speech_samples"]
    likelihoods = vad["likelihoods"]
    redemption_time = rust_duration_to_seconds(data["config"]["redemption_time"])
    pre_speech_pad = rust_duration_to_seconds(data["config"]["pre_speech_pad"])

    positive_thresh = float(data["config"]["positive_speech_threshold"]) * 100
    negative_thresh = float(data["config"]["negative_speech_threshold"]) * 100

    print(f"redemption time: {redemption_time}")
    for segment in vad["transitions"]:
        if "SpeechStart" in segment:
            start = int(segment["SpeechStart"]["timestamp_ms"])
            end = None
        elif "SpeechEnd" in segment:
            end = int(segment["SpeechEnd"]["end_timestamp_ms"])
            if start is not None and end is not None:
                speech_segments.append((start, end))
                start = None
                end = None


sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()
t_audio = n_samples/sample_freq

redemption_time_samples = sample_freq * redemption_time

duration = 1000*n_samples / sample_freq
finished_with_silence = True
if start is not None and end is None:
    finished_with_silence = False
    end = int(round(duration))
    speech_segments.append((start, end))

print(f"Segments: {speech_segments}")

signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)


times = np.linspace(0, n_samples/sample_freq, num=n_samples)

fig, (ax, ax2, ax3) = plt.subplots(3)

ax.plot(times, signal_array)

ax.set(xlabel="Time (s)", ylabel="Signal", title="Audio")
ax2.set(title = "Buffer Sizes")
ax3.set(title = "Network likelihoods")

ax2.plot(silence_samples, label = "Current silence samples")
ax2.plot(speech_samples, label = "Current speech samples")
ax3.plot(likelihoods, label = "network likelihoods")
labeled_start = False
labeled_end = False
for (i, (start, end)) in enumerate(speech_segments):
    if start > 0:
        ax.axvline(
            start/1000+pre_speech_pad,
            ymin=min(signal_array),
            ymax=max(signal_array),
            linestyle='dashed',
            color = '#42f5c2',
            label=None if labeled_start else "speech_start"
        )
        labeled_start = True
    if not finished_with_silence and i==len(speech_segments)-1:
        break 
    ax.axvline(
        end/1000+redemption_time,
        ymin=min(signal_array),
        ymax=max(signal_array),
        linestyle='dashed',
        color = '#f56f42',
        label=None if labeled_end else "speech_end"
    )
    labeled_end = True
ax.legend()

ax2.axhline(y=redemption_time_samples, color = 'r', linestyle = 'dashed', label = "redemption_time")
ax2.legend()

ax3.axhline(y=positive_thresh, color = 'g', linestyle = 'dashed', label = "positive threshold")
ax3.axhline(y=negative_thresh, color = 'r', linestyle = 'dashed', label = "negative threshold")
ax3.legend()

fill_regions = [False] * len(signal_array)

for i in range(len(signal_array)):
    timestamp =  1000*i/sample_freq

    for (start, end) in speech_segments:
        if start <= timestamp < end:
            fill_regions[i] = True
            break

ax.fill_between(times, min(signal_array), max(signal_array), where=fill_regions, alpha=0.5)

if args.output:
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
else:
    plt.show()

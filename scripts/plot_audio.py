import wave
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio", "-a", help="audio file to plot")
parser.add_argument("--input", "-i", help="test data file to plot")

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
    redemption_time = rust_duration_to_seconds(data["config"]["redemption_time"])
    print(f"redemption time: {redemption_time}")
    for segment in vad["transitions"]:
        if "SpeechStart" in segment:
            start = int(segment["SpeechStart"]["timestamp_ms"])
            end = None
        elif "SpeechEnd" in segment:
            end = int(segment["SpeechEnd"]["timestamp_ms"])
            if start is not None and end is not None:
                speech_segments.append((start, end))
                start = None
                end = None


sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()
t_audio = n_samples/sample_freq

redemption_time = sample_freq * redemption_time

duration = 1000*n_samples / sample_freq
if start is not None and end is None:
    end = int(round(duration))
    speech_segments.append((start, end))

print(f"Segments: {speech_segments}")

signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)


times = np.linspace(0, n_samples/sample_freq, num=n_samples)

fig, (ax, ax2) = plt.subplots(2)

ax.plot(times, signal_array)

ax.set(xlabel="Time (s)", ylabel="Signal", title="Audio")

ax2.plot(silence_samples, label = "Current silence samples")
ax2.plot(speech_samples, label = "Current speech samples")
ax2.axhline(y=redemption_time, color = 'r', linestyle = 'dashed', label = "redemption_time")
ax2.legend()

fill_regions = [False] * len(signal_array)

for i in range(len(signal_array)):
    timestamp =  1000*i/sample_freq

    for (start, end) in speech_segments:
        if start <= timestamp < end:
            fill_regions[i] = True
            break

ax.fill_between(times, min(signal_array), max(signal_array), where=fill_regions, alpha=0.5)

plt.show()

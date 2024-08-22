import wave
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio", "-a", help="audio file to plot")
parser.add_argument("--vad", "-v", help="vad output json to plot")

args = parser.parse_args() 


wav_obj = wave.open(args.audio, 'rb')

speech_segments = []

padding_samples = 0
with open(args.vad) as f:
    vad = json.load(f)
    padding_samples = int(vad["padding_samples"])
    for segment in vad["segments"]:
        speech_segments.append((segment["start_ms"], segment["end_ms"]))


sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes() + padding_samples
t_audio = n_samples/sample_freq

signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)

if padding_samples > 0:
    padding = np.zeros(padding_samples, dtype=np.int16)
    signal_array = np.concatenate((signal_array, padding ))


times = np.linspace(0, n_samples/sample_freq, num=n_samples)

fig, ax = plt.subplots()

ax.plot(times, signal_array)

ax.set(xlabel="Time (s)", ylabel="Signal", title="Audio")


fill_regions = [False] * len(signal_array)

has_printed = False

for i in range(len(signal_array)):
    timestamp =  1000*i/sample_freq

    for (start, end) in speech_segments:
        if start <= timestamp < end:
            fill_regions[i] = True
            break

ax.fill_between(times, min(signal_array), max(signal_array), where=fill_regions, alpha=0.5)

plt.show()

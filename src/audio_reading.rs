//! Some useful utilities to handle audio file reading, resampling, and writing. Should work with
//! any audio format.

// System libraries.
use std::fs::File;
use std::path::Path;

// Third party libraries.
use anyhow::Result;
use hound::{WavSpec, WavWriter};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::{debug, info};

// Project libraries.
use crate::utils::resample_pcm;

/// The way to deal with Stereo audio file
pub enum MultiChannelStrategy {
    /// Only take the first channel data for VAD inference.
    FirstOnly,

    /// Take the average of all channels for VAD inference.
    Average,
}

/// Read an audio file of arbitrary format into pcm data with desired sampling rate.
/// Code modified from https://github.com/pdeljanov/Symphonia/blob/master/symphonia/examples/basic-interleaved.rs
pub fn read_audio(
    file_path: impl AsRef<Path>,
    desired_sample_rate: usize,
    multi_channel_strategy: MultiChannelStrategy,
) -> Vec<f32> {
    info!("Reading {}", file_path.as_ref().display());
    // Create a media source. Note that the MediaSource trait is automatically implemented for File,
    // among other types.
    let file = Box::new(File::open(file_path).unwrap());

    // Create the media source stream using the boxed media source from above.
    let mss = MediaSourceStream::new(file, Default::default());

    // Create a hint to help the format registry guess what format reader is appropriate. In this
    // example we'll leave it empty.
    let hint = Hint::new();

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .unwrap();

    // Get the format reader yielded by the probe operation.
    let mut format = probed.format;

    // Get the default track.
    let track = format.default_track().unwrap();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .unwrap();

    // Store the track identifier, we'll use it to filter packets.
    let track_id = track.id;

    let mut sample_buf = None;
    let mut interleave_samples = vec![];
    let mut original_sample_rate = 0;
    let mut num_of_channels = 0;

    loop {
        // Get the next packet from the format reader.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => {
                break;
            }
        };

        // If the packet does not belong to the selected track, skip it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // The decoded audio samples may now be accessed via the audio buffer if per-channel
                // slices of samples in their native decoded format is desired. Use-cases where
                // the samples need to be accessed in an interleaved order or converted into
                // another sample format, or a byte buffer is required, are covered by copying the
                // audio buffer into a sample buffer or raw sample buffer, respectively. In the
                // example below, we will copy the audio buffer into a sample buffer in an
                // interleaved order while also converting to a f32 sample format.

                // If this is the *first* decoded packet, create a sample buffer matching the
                // decoded audio buffer format.
                if sample_buf.is_none() {
                    // Get the audio buffer specification.
                    let spec = *audio_buf.spec();
                    original_sample_rate = spec.rate;
                    num_of_channels = spec.channels.count();
                    debug!(
                        "Original file has {} channel(s) with sample rate {}",
                        num_of_channels, original_sample_rate
                    );

                    // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                    let duration = audio_buf.capacity() as u64;

                    // Create the f32 sample buffer.
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);
                    interleave_samples.extend_from_slice(buf.samples());
                }
            }
            Err(Error::DecodeError(_)) => (),
            Err(Error::IoError(e)) => {
                dbg!(e);
                break;
            }
            Err(_) => break,
        }
    }

    // Now we have interleaved audio, need to convert it to channel based.
    let stereo_audio = interleaved_to_channel(interleave_samples, num_of_channels);

    // Silero VAD only work with mono audio.
    let mono_audio = stereo_to_mono(stereo_audio, multi_channel_strategy);

    // Silero VAD only work with 8000 or 16000 Hz audio.
    resample_pcm(
        mono_audio,
        original_sample_rate as usize,
        desired_sample_rate,
    )
    .unwrap()
}

/// Convert stereo audio to mono audio based on [MultiChannelStrategy].
fn stereo_to_mono(stereo: Vec<Vec<f32>>, multi_channel_strategy: MultiChannelStrategy) -> Vec<f32> {
    match multi_channel_strategy {
        MultiChannelStrategy::FirstOnly => stereo[0].clone(),
        MultiChannelStrategy::Average => {
            let samples = stereo[0].len();
            let channels = stereo.len();
            let mut mono = Vec::new();
            mono.resize(samples, 0.0);

            for i in 0..samples {
                let mut sum = 0.0;
                for channel in 0..channels {
                    sum += stereo[channel][i];
                }
                mono[i] = sum / channels as f32;
            }
            mono
        }
    }
}

/// Convert an interleaved pcm data to channel based.
fn interleaved_to_channel(interleave_samples: Vec<f32>, num_of_channels: usize) -> Vec<Vec<f32>> {
    let mut audio = vec![vec![]; num_of_channels];
    let mut channel_idx = 0;
    for sample in interleave_samples {
        audio[channel_idx].push(sample);
        channel_idx += 1;
        channel_idx %= num_of_channels;
    }

    // A quick sanity check
    let len = audio[0].len();
    for channel in &audio {
        assert_eq!(len, channel.len());
    }

    audio
}

/// Save a mono pcm data into wav file. This method is only for debugging.
pub fn save_pcm_to_wav(
    file_path: &str,
    pcm_data: Vec<f32>,
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    // Define the WAV file specifications
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Create a WAV writer
    let mut writer = WavWriter::create(file_path, spec)?;

    // Write PCM data to the WAV file, converting f32 samples to i16
    for sample in pcm_data {
        // Convert f32 sample to i16
        let sample_i16 = (sample * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }

    // Finalize the WAV file
    writer.finalize()?;

    Ok(())
}

/// Convert to i16 to satisfy VAD model requirement.
pub fn vec_f32_to_vec_i16(data: &[f32]) -> Vec<i16> {
    data.iter()
        .map(|&sample| {
            // Clamp the values to the -1.0 to 1.0 range to prevent unexpected behavior
            let clamped_sample = sample.clamp(-1.0, 1.0);
            // Scale the clamped sample to the i16 range
            let scaled_sample = clamped_sample * 32767.0;
            // Convert to i16
            scaled_sample as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Should be able to read wav file.
    #[test]
    fn read_wav() {
        let file_path = "./tests/audio/sample_1.wav";
        let pcm_data = read_audio(file_path, 16000, MultiChannelStrategy::FirstOnly);

        assert!(pcm_data.len() > 0);
    }

    /// Should be able to read mp3 file.
    #[test]
    fn read_mp3() {
        let file_path = "./tests/audio/sample_1.mp3";
        let pcm_data = read_audio(file_path, 16000, MultiChannelStrategy::FirstOnly);

        assert!(pcm_data.len() > 0);
    }

    /// Should be able to read audio file into desired sample rate.
    #[test]
    fn read_audio_with_desired_sample_rate() {
        let wav_file_path = "./tests/audio/sample_1.wav";
        let mp3_file_path = "./tests/audio/sample_1.mp3";

        // Original audio file is 16k Hz.
        let pcm_from_wav_16k_hz = read_audio(wav_file_path, 16000, MultiChannelStrategy::FirstOnly);
        let pcm_from_mp3_16k_hz = read_audio(mp3_file_path, 16000, MultiChannelStrategy::FirstOnly);
        assert!(pcm_from_wav_16k_hz.len() > 0);
        assert!(pcm_from_mp3_16k_hz.len() > 0);
        // The online converter I used to generate mp3 file might add some pads to it, so cannot
        // check pcm_from_wav_16k_hz.len() == pcm_from_mp3_16k_hz.len()

        // Down-sample to 8k Hz.
        let pcm_from_wav_8k_hz = read_audio(wav_file_path, 8000, MultiChannelStrategy::FirstOnly);
        let pcm_from_mp3_8k_hz = read_audio(mp3_file_path, 8000, MultiChannelStrategy::FirstOnly);
        assert!(pcm_from_wav_8k_hz.len() > 0);
        assert!(pcm_from_mp3_8k_hz.len() > 0);

        assert_eq!(pcm_from_wav_16k_hz.len() / 2, pcm_from_wav_8k_hz.len());
        assert_eq!(pcm_from_mp3_16k_hz.len() / 2, pcm_from_mp3_8k_hz.len());

        // Up-sample to 32k Hz.
        let pcm_from_wav_32k_hz = read_audio(wav_file_path, 32000, MultiChannelStrategy::FirstOnly);
        let pcm_from_mp3_32k_hz = read_audio(mp3_file_path, 32000, MultiChannelStrategy::FirstOnly);
        assert!(pcm_from_wav_32k_hz.len() > 0);
        assert!(pcm_from_mp3_32k_hz.len() > 0);

        assert_eq!(pcm_from_wav_16k_hz.len() * 2, pcm_from_wav_32k_hz.len());
        assert_eq!(pcm_from_mp3_16k_hz.len() * 2, pcm_from_mp3_32k_hz.len());
    }
}

//! Some useful utilities to handle audio file reading, resampling, and writing. Should work with
//! any audio format.

// System libraries.
use std::fs::File;
use std::path::Path;

// Third party libraries.
use anyhow::Result;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::{debug, info};

// Project libraries.
use silero::audio_resampler::resample_pcm;

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

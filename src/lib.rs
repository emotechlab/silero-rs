#![doc = include_str!("../README.md")]
#[cfg(feature = "audio_resampler")]
pub use crate::audio_resampler::resample_pcm;
pub use crate::errors::VadError;
use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::ops::Range;
use std::path::Path;
use std::time::Duration;
use tracing::trace;

#[cfg(feature = "audio_resampler")]
pub mod audio_resampler;
pub mod errors;

/// Parameters used to configure a vad session. These will determine the sensitivity and switching
/// speed of detection.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub pre_speech_pad: Duration,
    pub post_speech_pad: Duration,
    pub redemption_time: Duration,
    pub sample_rate: usize,
    pub min_speech_time: Duration,
}

/// A VAD session create one of these for each audio stream you want to detect voice activity on
/// and feed the audio into it.
#[derive(Debug)]
pub struct VadSession {
    config: VadConfig,
    model: Session, // TODO: would this be safe to share? does the runtime graph hold any state?
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
    state: VadState,
    session_audio: Vec<f32>,
    processed_samples: usize,
    deleted_samples: usize,
    silent_samples: usize,

    /// Current start of the speech in milliseconds
    speech_start_ms: Option<usize>,

    /// Cached current active samples
    cached_active_speech: Vec<f32>,
}

/// Current state of the VAD (speaking or silent)
#[derive(Clone, Debug)]
enum VadState {
    Speech {
        start_ms: usize,
        redemption_passed: bool,
        speech_time: Duration,
    },
    Silence,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VadTransition {
    SpeechStart {
        /// When the speech started, in milliseconds since the start of the VAD session.
        timestamp_ms: usize,
    },
    SpeechEnd {
        /// When the speech started, in milliseconds since the start of the VAD session.
        start_timestamp_ms: usize,
        /// When the speech ended, in milliseconds since the start of the VAD session.
        end_timestamp_ms: usize,
        /// The active speech samples. This field is skipped in serde output even serde feature is enabled.
        #[cfg_attr(feature = "serde", serde(default, skip))]
        samples: Vec<f32>,
    },
}

impl PartialEq for VadTransition {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                VadTransition::SpeechStart { timestamp_ms: ts1 },
                VadTransition::SpeechStart { timestamp_ms: ts2 },
            ) => ts1 == ts2,
            (
                VadTransition::SpeechEnd {
                    start_timestamp_ms: ts1,
                    end_timestamp_ms: ts2,
                    ..
                },
                VadTransition::SpeechEnd {
                    start_timestamp_ms: ts3,
                    end_timestamp_ms: ts4,
                    ..
                },
            ) => ts1 == ts3 && ts2 == ts4,
            _ => false,
        }
    }
}

impl Eq for VadTransition {}

impl VadSession {
    /// Create a new VAD session loading an onnx file from the specified path and using the
    /// provided config.
    pub fn new_from_path(file: impl AsRef<Path>, config: VadConfig) -> Result<Self> {
        let bytes = std::fs::read(file.as_ref())
            .with_context(|| format!("Couldn't read onnx file: {}", file.as_ref().display()))?;
        Self::new_from_bytes(&bytes, config)
    }

    /// Create a new VAD session loading an onnx file from memory and using the provided config.
    pub fn new_from_bytes(model_bytes: &[u8], config: VadConfig) -> Result<Self> {
        if ![8000_usize, 16000].contains(&config.sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_memory(model_bytes)?;
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));
        let sample_rate_tensor = Array1::from_vec(vec![config.sample_rate as i64]);

        Ok(Self {
            config,
            model,
            h_tensor,
            c_tensor,
            sample_rate_tensor,
            state: VadState::Silence,
            session_audio: vec![],
            processed_samples: 0,
            deleted_samples: 0,
            silent_samples: 0,
            speech_start_ms: None,
            cached_active_speech: vec![],
        })
    }

    /// Create a new VAD session using the provided config. The ONNX file has been statically
    /// embedded within the library so this will increase binary size by 1.7M.
    #[cfg(feature = "static-model")]
    pub fn new(config: VadConfig) -> Result<Self> {
        let model_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/silero_vad.onnx"
        ));
        Self::new_from_bytes(model_bytes, config)
    }

    pub fn validate_input(&self, audio_frame: &[f32]) -> Result<()> {
        if audio_frame
            .iter()
            .all(|&sample| -1.0 <= sample && sample <= 1.0)
        {
            Ok(())
        } else {
            Err(VadError::InvalidData.into())
        }
    }

    /// Pass in some audio to the VAD and return a list of any speech transitions that happened
    /// during the segment.
    ///
    /// If "audio_resampler" feature is enabled, then this function will resample whatever it
    /// receives to 16000Hz samples. If not enabled, then user input is expected to be either
    /// 8000 or 16000Hz.
    pub fn process(&mut self, audio_frame: &[f32]) -> Result<Vec<VadTransition>> {
        #[cfg(debug_assertions)]
        if let Err(e) = self.validate_input(audio_frame) {
            return Err(e);
        }

        #[cfg(feature = "audio_resampler")]
        let audio_frame = if ![8000, 16000].contains(&self.config.sample_rate) {
            &resample_pcm(audio_frame.to_vec(), self.config.sample_rate, 16000)?
        } else {
            audio_frame
        };

        const VAD_BUFFER: Duration = Duration::from_millis(30); // TODO This should be configurable
        let vad_segment_length = VAD_BUFFER.as_millis() as usize * self.config.sample_rate / 1000;

        let unprocessed = self.deleted_samples + self.session_audio.len() - self.processed_samples;
        let num_chunks = (unprocessed + audio_frame.len()) / vad_segment_length;

        self.session_audio.extend_from_slice(audio_frame);

        let mut transitions = vec![];

        for i in 0..num_chunks {
            // we might not be getting audio chunks in perfect multiples of 30ms, so let the
            // last frame accommodate the remainder. This adds a bit of non-determinism based on
            // audio size but it does let us more eagerly process audio.
            //
            // processed_samples is updated in process_internal so always points to the index of
            // the next sample to go from.
            let sample_range = if i < num_chunks - 1 {
                (self.processed_samples - self.deleted_samples)
                    ..(self.processed_samples + vad_segment_length - self.deleted_samples)
            } else {
                (self.processed_samples - self.deleted_samples)..self.session_audio.len()
            };
            let vad_result = self.process_internal(sample_range)?;

            if let Some(vad_ev) = vad_result {
                transitions.push(vad_ev);
            }
        }
        Ok(transitions)
    }

    pub fn forward(&mut self, input: Vec<f32>) -> Result<ort::value::Value> {
        let samples = input.len();
        let audio_tensor = Array2::from_shape_vec((1, samples), input)?;
        let mut result = self.model.run(ort::inputs![
            audio_tensor.view(),
            self.sample_rate_tensor.view(),
            self.h_tensor.view(),
            self.c_tensor.view()
        ]?)?;

        // Update internal state tensors.
        self.h_tensor = result
            .get("hn")
            .unwrap()
            .try_extract_tensor::<f32>()?
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .context("Shape mismatch for h_tensor")?;

        self.c_tensor = result
            .get("cn")
            .unwrap()
            .try_extract_tensor::<f32>()?
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .context("Shape mismatch for h_tensor")?;

        let prob_tensor = result.remove("output").unwrap();
        Ok(prob_tensor)
    }

    /// Advance the VAD state machine with an audio frame. Keep between 30-96ms in length.
    /// Return indicates if a transition from speech to silence (or silence to speech) occurred.
    ///
    /// Important: don't implement your own endpointing logic.
    /// Instead, when a `SpeechEnd` is returned, you can use the `get_current_speech()` method to retrieve the audio.
    fn process_internal(&mut self, range: Range<usize>) -> Result<Option<VadTransition>> {
        let audio_frame = self.session_audio[range].to_vec();
        let samples = audio_frame.len();
        let frame_duration = self.samples_to_duration(samples);

        let result = self.forward(audio_frame)?;

        let prob = *result.try_extract_tensor::<f32>().unwrap().first().unwrap();

        let mut vad_change = None;

        if prob < self.config.negative_speech_threshold {
            self.silent_samples += samples;
        } else {
            self.silent_samples = 0;
        }

        trace!(
            vad_likelihood = prob,
            samples,
            silent_samples = self.silent_samples,
            "performed silero inference"
        );

        let current_silence = self.current_silence_duration();

        match self.state {
            VadState::Silence => {
                if prob > self.config.positive_speech_threshold {
                    self.state = VadState::Speech {
                        start_ms: self
                            .processed_duration()
                            .saturating_sub(self.config.pre_speech_pad)
                            .as_millis() as usize,
                        redemption_passed: false,
                        speech_time: Duration::ZERO,
                    };
                }
            }
            VadState::Speech {
                start_ms,
                ref mut redemption_passed,
                ref mut speech_time,
            } => {
                *speech_time += frame_duration;
                if !*redemption_passed && *speech_time > self.config.min_speech_time {
                    *redemption_passed = true;
                    // TODO: the pre speech padding should not cross over the previous speech->silence
                    // transition, if there was one
                    vad_change = Some(VadTransition::SpeechStart {
                        timestamp_ms: start_ms,
                    });
                    self.speech_start_ms = Some(start_ms);
                }

                if prob < self.config.negative_speech_threshold {
                    if !*redemption_passed {
                        self.state = VadState::Silence;
                    } else if current_silence > self.config.redemption_time {
                        if *redemption_passed {
                            let speech_end_ms = (self.processed_samples + samples
                                - self.silent_samples)
                                / (self.config.sample_rate / 1000);

                            // Since speech_end_ms does not include silent_samples, the post
                            // padding should be added to speech_end_ms.
                            let speech_end_with_pad_ms =
                                speech_end_ms + self.config.post_speech_pad.as_millis() as usize;
                            // TODO: Just some quick double checks, remember to delete these before git merge.
                            assert!(
                                self.config.post_speech_pad.as_millis() as usize
                                    <= self.silent_samples
                            );
                            assert!(
                                self.duration_to_index(Duration::from_millis(
                                    speech_end_with_pad_ms as u64
                                ))
                                .unwrap()
                                    < self.session_audio.len()
                            );

                            vad_change = Some(VadTransition::SpeechEnd {
                                start_timestamp_ms: start_ms,
                                end_timestamp_ms: speech_end_ms,
                                samples: self
                                    .get_speech(start_ms, Some(speech_end_with_pad_ms))
                                    .to_vec(),
                            });

                            // Need to delete the current speech samples from internal buffer to prevent OOM.
                            assert!(self.speech_start_ms.is_some());
                            self.cached_active_speech = self
                                .get_speech(start_ms, Some(speech_end_with_pad_ms))
                                .to_vec();
                            let speech_end_with_pad_idx = self.unchecked_duration_to_index(
                                Duration::from_millis(speech_end_with_pad_ms as u64),
                            );
                            let to_delete_idx = 0..(speech_end_with_pad_idx + 1);
                            self.session_audio.drain(to_delete_idx);
                            self.deleted_samples += speech_end_with_pad_idx + 1;
                            self.speech_start_ms = None;
                        }
                        self.state = VadState::Silence
                    }
                }
            }
        };

        self.processed_samples += samples;

        Ok(vad_change)
    }

    /// This will remove audio in the buffer until a duration and panic if it exceeds the duration.
    /// This won't touch the active speech cache or the VAD state. It's intended usage is if the
    /// current speech buffer is too long and you want to remove some for processing and not have
    /// it be re-processed or considered again.
    ///
    /// If there is no remaining audio within the range this will return an empty vector.
    /// Additionally, if the speech is just in the cached last-segment it won't take from that
    /// (though this could be done in the future).
    ///
    /// # Panics
    ///
    /// If the time given is beyond the range of the current session this will panic.
    pub fn take_until(&mut self, end: Duration) -> Vec<f32> {
        if end > self.session_time() {
            panic!(
                "{}ms is greater than session time of {}ms",
                end.as_millis(),
                self.session_time().as_millis()
            );
        } else {
            match self.duration_to_index(end) {
                Some(s) => {
                    let mut returned_audio = self.session_audio.split_off(s);
                    std::mem::swap(&mut self.session_audio, &mut returned_audio);
                    self.deleted_samples += returned_audio.len();
                    if matches!(self.state, VadState::Speech { .. }) {
                        if let Some(start_ms) = self.speech_start_ms.take() {
                            if start_ms < end.as_millis() as usize {
                                self.speech_start_ms = Some(end.as_millis() as usize);
                            }
                        }
                    }
                    returned_audio
                }
                None => vec![],
            }
        }
    }

    /// Returns whether the vad current believes the audio to contain speech
    pub fn is_speaking(&self) -> bool {
        matches!(self.state, VadState::Speech {
            redemption_passed, ..
        } if redemption_passed)
    }

    /// Takes a duration and converts it to an index if it's within the current session audio or
    /// `None` if it's not.
    ///
    /// # Panics
    ///
    /// If this is out of the range it will panic
    fn unchecked_duration_to_index(&self, duration: Duration) -> usize {
        match self.duration_to_index(duration) {
            Some(idx) => idx,
            None => panic!(
                "Duration {}ms is outside of session audio range",
                duration.as_millis()
            ),
        }
    }

    /// Takes a duration and converts it to an index if it's within the current session audio or
    /// `None` if it's not.
    fn duration_to_index(&self, duration: Duration) -> Option<usize> {
        let unadjusted_index = duration.as_millis() as usize * (self.config.sample_rate / 1000);
        if unadjusted_index < self.deleted_samples {
            None
        } else {
            Some(unadjusted_index - self.deleted_samples)
        }
    }

    /// Gets the speech within a given range of milliseconds. You can use previous speech start/end
    /// event pairs to get speech windows before the current speech using this API. If end is
    /// `None` this will return from the start point to the end of the buffer.
    ///
    /// # Panics
    ///
    /// If the range is out of bounds of the speech buffer this method will panic due to an
    /// assertion failure.
    pub fn get_speech(&self, start_ms: usize, end_ms: Option<usize>) -> &[f32] {
        let speech_start_idx =
            self.unchecked_duration_to_index(Duration::from_millis(start_ms as u64));
        if let Some(speech_end) = end_ms {
            let speech_end_idx =
                self.unchecked_duration_to_index(Duration::from_millis(speech_end as u64));
            &self.session_audio[speech_start_idx..speech_end_idx]
        } else {
            &self.session_audio[speech_start_idx..]
        }
    }

    /// Gets a buffer of the most recent active speech frames from the time the speech started to the
    /// end of the speech. Parameters from `VadConfig` have already been applied here so this isn't
    /// derived from the raw VAD inferences but instead after padding and filtering operations have
    /// been applied.
    pub fn get_current_speech(&self) -> &[f32] {
        if let Some(speech_start) = self.speech_start_ms {
            self.get_speech(speech_start, None)
        } else {
            &self.cached_active_speech
        }
    }

    /// Get how long the current speech is in samples.
    pub fn current_speech_samples(&self) -> usize {
        self.get_current_speech().len()
    }

    /// Returns the duration of the current speech segment. It is possible for this and
    /// `Self::current_silence_duration` to both report >0s at  the same time as this takes into
    /// account the switching and padding parameters of the VAD whereas the silence measure ignores
    /// them instead of just focusing on raw network output.
    pub fn current_speech_duration(&self) -> Duration {
        self.samples_to_duration(self.current_speech_samples())
    }

    /// Get the current duration of the VAD session, which includes both processed and unprocessed
    /// samples.
    pub fn session_time(&self) -> Duration {
        self.samples_to_duration(self.session_audio.len() + self.deleted_samples)
    }

    /// Get the current duration of processed samples. A sample is considered as processed if it has
    /// been seen by Silero neural network.
    pub fn processed_duration(&self) -> Duration {
        self.samples_to_duration(self.processed_samples)
    }

    /// Reset the status of the model
    // TODO should this reset the audio buffer as well?
    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.speech_start_ms = None;
        self.silent_samples = 0;
        self.state = VadState::Silence;
    }

    /// Returns the length of the end silence in number of samples. The VAD may be showing this as
    /// speaking because of redemption frames or other parameters that slow down the speed it can
    /// switch at. But this measure is a raw unprocessed look of how many segments since the last
    /// speech are below the negative speech threshold.
    pub fn current_silence_samples(&self) -> usize {
        self.silent_samples
    }

    /// Returns the duration of the end silence. The VAD may be showing this as speaking because of
    /// redemption frames or other parameters that slow down the speed it can switch at. But this
    /// measure is a raw unprocessed look of how many segments since the last speech are below the
    /// negative speech threshold.
    pub fn current_silence_duration(&self) -> Duration {
        self.samples_to_duration(self.silent_samples)
    }

    /// Returns an inclusive range of the audio currently stored in the session buffer. The
    /// previously complete active speech segment may exceed these bounds!
    pub fn current_buffer_range(&self) -> (Duration, Duration) {
        (
            self.samples_to_duration(self.deleted_samples),
            self.session_time(),
        )
    }

    #[inline(always)]
    fn samples_to_duration(&self, samples: usize) -> Duration {
        Duration::from_secs_f64(samples as f64 / self.config.sample_rate as f64)
    }

    /// Utility function to add a bit more tracking into the snapshot tests
    #[doc(hidden)]
    pub fn session_audio_samples(&self) -> usize {
        self.session_audio.len()
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            // https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
            positive_speech_threshold: 0.5,
            negative_speech_threshold: 0.35,
            pre_speech_pad: Duration::from_millis(600),
            post_speech_pad: Duration::from_millis(0),
            redemption_time: Duration::from_millis(600),
            sample_rate: 16000,
            min_speech_time: Duration::from_millis(90),
        }
    }
}

impl VadConfig {
    pub fn new(
        positive_speech_threshold: f32,
        negative_speech_threshold: f32,
        pre_speech_pad: Duration,
        post_speech_pad: Duration,
        redemption_time: Duration,
        sample_rate: usize,
        min_speech_time: Duration,
    ) -> Result<Self> {
        let config = VadConfig {
            positive_speech_threshold,
            negative_speech_threshold,
            pre_speech_pad,
            post_speech_pad,
            redemption_time,
            sample_rate,
            min_speech_time,
        };
        match config.validate_config() {
            Ok(_) => Ok(config),
            Err(e) => Err(e),
        }
    }

    pub fn validate_config(&self) -> Result<()> {
        if self.post_speech_pad > self.redemption_time {
            bail!("post speech pad cannot be longer than redemption time, as it's illogical")
        }

        #[cfg(feature = "audio_resampler")]
        return Ok(());

        #[cfg(not(feature = "audio_resampler"))]
        if ![8000, 16000].contains(&self.sample_rate) {
            bail!(
                "Invalid sample rate of {}, expected either 8000Hz or 16000Hz",
                self.sample_rate
            );
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;

    /// Feed only silence into the network and ensure that `get_current_speech` returns an empty
    /// slice
    #[test]
    #[traced_test]
    fn only_silence_get_speech() {
        let mut session = VadSession::new(VadConfig::default()).unwrap();
        let short_audio = vec![0.0; 1000];

        session.process(&short_audio).unwrap();

        assert_eq!(session.get_current_speech(), &[]);
    }

    /// Basic smoke test that the model loads correctly and we haven't committed rubbish to the
    /// repo.
    #[test]
    #[traced_test]
    fn model_loads() {
        let _session = VadSession::new(VadConfig::default()).unwrap();
        let _session =
            VadSession::new_from_path("models/silero_vad.onnx", VadConfig::default()).unwrap();
    }

    /// Too short tensors result in inference errors which we don't want to unnecessarily bubble up
    /// to the user and instead handle in our buffering implementation. This test will check that a
    /// short inference in the internal inference call bubbles up an error but when using the
    /// public API no error is presented.
    #[test]
    #[traced_test]
    fn short_audio_handling() {
        let mut session = VadSession::new(VadConfig::default()).unwrap();

        let short_audio = vec![0.0; 160];

        session.session_audio = short_audio.clone();
        assert!(session.process_internal(0..160).is_err());
        session.session_audio.clear();
        assert!(session.process(&short_audio).unwrap().is_empty());
    }

    /// Check that a long enough packet of just zeros gets an inference and it doesn't flag as
    /// transitioning to speech
    #[test]
    #[traced_test]
    fn silence_handling() {
        let mut session = VadSession::new(VadConfig::default()).unwrap();
        let silence = vec![0.0; 30 * 16]; // 30ms of silence

        assert!(session.process(&silence).unwrap().is_empty());
        assert_eq!(session.processed_samples, silence.len());
    }

    /// We only allow for 8khz and 16khz audio.
    #[test]
    #[traced_test]
    fn reject_invalid_sample_rate() {
        let mut config = VadConfig::default();
        config.sample_rate = 16000;
        VadSession::new(config.clone()).unwrap();
        config.sample_rate = 8000;
        VadSession::new(config.clone()).unwrap();

        config.sample_rate += 1;
        assert!(VadSession::new(config.clone()).is_err());
        assert!(VadSession::new_from_path("models/silero_vad.onnx", config.clone()).is_err());

        let bytes = std::fs::read("models/silero_vad.onnx").unwrap();
        assert!(VadSession::new_from_bytes(&bytes, config).is_err());
    }

    /// Just a sanity test of speech duration to make sure the calculation seems roughly right in
    /// terms of number of samples, sample rate and taking into account the speech starts/ends.
    #[test]
    #[traced_test]
    fn simple_speech_duration() {
        let mut config = VadConfig::default();
        config.sample_rate = 8000;
        let mut session = VadSession::new(config.clone()).unwrap();
        session.session_audio.resize(16080, 0.0);

        assert_eq!(session.current_speech_duration(), Duration::from_secs(0));

        session.speech_start_ms = Some(10);
        assert_eq!(session.current_speech_duration(), Duration::from_secs(2));

        session.config.sample_rate = 16000;
        session.session_audio.resize(16160, 0.0);
        assert_eq!(session.current_speech_duration(), Duration::from_secs(1));
    }

    /// The provided audio sample must be in the range -1.0 to 1.0
    #[test]
    #[traced_test]
    fn audio_sample_range() {
        let config = VadConfig::default();

        let mut session = VadSession::new(config.clone()).unwrap();
        let valid_samples = [0.0; 1000];
        let result = session.process(&valid_samples);
        assert!(result.is_ok());

        let mut session2 = VadSession::new(config).unwrap();
        let mut invalid_samples = valid_samples.clone();
        invalid_samples[0] = -1.01;
        let result = session2.process(&invalid_samples);
        assert!(matches!(
            result.unwrap_err().downcast::<VadError>().unwrap(),
            VadError::InvalidData
        ));
    }

    /// Apply some audio with speech in and ensure that the take API works as expected
    #[test]
    #[traced_test]
    fn taking_audio() {
        let samples: Vec<f32> = hound::WavReader::open("tests/audio/sample_4.wav")
            .unwrap()
            .into_samples()
            .map(|x| {
                let modified = x.unwrap_or(0i16) as f32 / (i16::MAX as f32);
                modified.clamp(-1.0, 1.0)
            })
            .collect();

        let config = VadConfig::default();
        let mut session = VadSession::new(config.clone()).unwrap();

        let chunk_size = 480; // 30ms
        let max_chunks = samples.len() / chunk_size;

        let mut start_time = 0;

        for i in 0..max_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(samples.len());
            let trans = session.process(&samples[start..end]).unwrap();

            for transition in &trans {
                match transition {
                    VadTransition::SpeechStart { timestamp_ms } => {
                        start_time = *timestamp_ms as u64;
                    }
                    _ => panic!("Oh no it's over"),
                }
            }

            if session.is_speaking()
                && (session.session_time() - Duration::from_millis(start_time))
                    >= Duration::from_millis(120)
            {
                break;
            }
        }
        assert!(session.is_speaking(), "never found speech");

        let current_untaken = session.get_current_speech().to_vec();

        let session_end = session.session_time();

        let (start, end) = session.current_buffer_range();
        assert_eq!(start, Duration::from_secs(0));
        assert_eq!(end, session_end);

        let until = Duration::from_millis(start_time + 60);
        let taken = session.take_until(until);

        let (start, end) = session.current_buffer_range();
        assert_eq!(start, until);
        assert_eq!(end, session_end);

        assert!(session.current_speech_samples() < current_untaken.len());
        assert_eq!(
            session.current_speech_samples() + taken.len(),
            current_untaken.len()
        );

        assert!(session
            .take_until(Duration::from_millis(start_time))
            .is_empty());
    }

    /// If we take past our boundary we panic!
    #[test]
    #[traced_test]
    #[should_panic]
    fn excessive_take() {
        let config = VadConfig::default();
        let mut session = VadSession::new(config.clone()).unwrap();

        let silence = vec![0.0; 16000];
        let _ = session.process(&silence);

        session.take_until(Duration::from_millis(1001));
    }
}

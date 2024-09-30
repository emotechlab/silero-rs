#![doc = include_str!("../README.md")]
use anyhow::{bail, Context, Result};
use ndarray::{s, Array, Array2, ArrayBase, ArrayD, Dim, OwnedRepr};
use ort::{GraphOptimizationLevel, Session, Tensor};
use std::ops::Range;
use std::path::Path;
use std::time::Duration;

/// Parameters used to configure a vad session. These will determine the sensitivity and switching
/// speed of detection.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub pre_speech_pad: Duration,
    pub redemption_time: Duration,
    pub sample_rate: usize,
    pub min_speech_time: Duration,
}

impl VadConfig {
    /// Gets the number of audio samples in an input frame
    pub fn get_frame_samples(&self) -> usize {
        (30_f32 / 1000_f32 * self.sample_rate as f32) as usize // 30ms * sample_rate Hz
    }

    /// Gets the number of frames for a given duration in milliseconds
    pub fn get_frames(length_ms: usize) -> usize {
        length_ms / 30
    }
}

/// A VAD session create one of these for each audio stream you want to detect voice activity on
/// and feed the audio into it.
#[derive(Debug)]
pub struct VadSession {
    config: VadConfig,
    model: Session, // TODO: would this be safe to share? does the runtime graph hold any state?
    state_tensor: ArrayD<f32>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    state: VadState,
    session_audio: Vec<f32>,
    processed_samples: usize,
    silent_samples: usize,
    speech_start: Option<usize>,
    speech_end: Option<usize>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VadTransition {
    SpeechStart {
        /// When the speech started, in milliseconds since the start of the VAD session.
        timestamp_ms: usize,
    },
    SpeechEnd {
        /// When the speech ended, in milliseconds since the start of the VAD session.
        timestamp_ms: usize,
    },
}

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
        let state_tensor = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        let sample_rate_tensor = Array::from_shape_vec([1], vec![config.sample_rate as i64]).unwrap();

        Ok(Self {
            config,
            model,
            state_tensor,
            sample_rate_tensor,
            state: VadState::Silence,
            session_audio: vec![],
            processed_samples: 0,
            silent_samples: 0,
            speech_start: None,
            speech_end: None,
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

    /// Pass in some audio to the VAD and return a list of any speech transitions that happened
    /// during the segment.
    pub fn process(&mut self, audio_frame: &[f32]) -> Result<Vec<VadTransition>> {
        const VAD_BUFFER_MS: usize = 30; // TODO This should be configurable
        let vad_segment_length = VAD_BUFFER_MS * self.config.sample_rate / 1000;

        let unprocessed = self.session_audio.len() - self.processed_samples;
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
                self.processed_samples..(self.processed_samples + vad_segment_length)
            } else {
                self.processed_samples..self.session_audio.len()
            };
            let vad_result = self.process_internal(sample_range)?;

            if let Some(vad_ev) = vad_result {
                transitions.push(vad_ev);
            }
        }
        Ok(transitions)
    }

    /// Advance the VAD state machine with an audio frame. Keep between 30-96ms in length.
    /// Return indicates if a transition from speech to silence (or silence to speech) occurred.
    ///
    /// Important: don't implement your own endpointing logic.
    /// Instead, when a `SpeechEnd` is returned, you can use the `get_current_speech()` method to retrieve the audio.
    fn process_internal(&mut self, range: Range<usize>) -> Result<Option<VadTransition>> {
        let audio_frame = &self.session_audio[range];
        let samples = audio_frame.len();
        // FIXME: handling of remainder frames
        let mut audio_tensor = Array2::<f32>::from_shape_vec([1, samples], audio_frame.to_vec())?;
        audio_tensor = match self.config.sample_rate {
            16000 => audio_tensor.slice(s![.., ..480.min(samples)]).to_owned(),
            8000 => audio_tensor.slice(s![.., ..240.min(samples)]).to_owned(),
            _ => unreachable!(),
        };

        let inputs = ort::inputs![
            audio_tensor,
            std::mem::take(&mut self.state_tensor),
            self.sample_rate_tensor.clone(),
        ]?;
        let result = self.model.run(ort::SessionInputs::ValueSlice::<3>(&inputs))?;

        self.state_tensor = result["stateN"].try_extract_tensor().unwrap().to_owned();

        let prob = *result["output"]
            .try_extract_raw_tensor::<f32>()
            .unwrap()
            .1
            .first()
            .unwrap();

        let mut vad_change = None;

        if prob < self.config.negative_speech_threshold {
            self.silent_samples += samples;
        } else {
            self.silent_samples = 0;
        }

        let current_silence = self.current_silence_duration();

        match self.state {
            VadState::Silence => {
                if prob > self.config.positive_speech_threshold {
                    self.state = VadState::Speech {
                        start_ms: self
                            .session_time()
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
                *speech_time +=
                    Duration::from_secs_f64(samples as f64 / self.config.sample_rate as f64);
                if !*redemption_passed && *speech_time > self.config.min_speech_time {
                    *redemption_passed = true;
                    // TODO: the pre speech padding should not cross over the previous speech->silence
                    // transition, if there was one
                    vad_change = Some(VadTransition::SpeechStart {
                        timestamp_ms: start_ms,
                    });
                    self.speech_start = Some(start_ms);
                    self.speech_end = None;
                }

                if prob < self.config.negative_speech_threshold {
                    if !*redemption_passed {
                        self.state = VadState::Silence;
                    } else {
                        if current_silence > self.config.redemption_time {
                            if *redemption_passed {
                                let speech_end = (self.processed_samples + audio_frame.len()
                                    - self.silent_samples)
                                    / (self.config.sample_rate / 1000);
                                vad_change = Some(VadTransition::SpeechEnd {
                                    timestamp_ms: speech_end,
                                });
                                self.speech_end = Some(speech_end);
                            }
                            self.state = VadState::Silence
                        }
                    }
                }
            }
        };

        self.processed_samples += audio_frame.len();

        Ok(vad_change)
    }

    /// Returns whether the vad current believes the audio to contain speech
    pub fn is_speaking(&self) -> bool {
        matches!(self.state, VadState::Speech {
            redemption_passed, ..
        } if redemption_passed)
    }

    /// Gets a buffer of the most recent active speech frames from the time the speech started to the
    /// end of the speech. Parameters from `VadConfig` have already been applied here so this isn't
    /// derived from the raw VAD inferences but instead after padding and filtering operations have
    /// been applied.
    pub fn get_current_speech(&self) -> &[f32] {
        if let Some(speech_start) = self.speech_start {
            let speech_start = speech_start * (self.config.sample_rate / 1000);
            if let Some(speech_end) = self.speech_end {
                let speech_end = speech_end * (self.config.sample_rate / 1000);
                &self.session_audio[speech_start..speech_end]
            } else {
                &self.session_audio[speech_start..]
            }
        } else {
            &[]
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
        Duration::from_millis(
            (self.current_speech_samples() / (self.config.sample_rate / 1000)) as u64,
        )
    }

    /// Get the current length of the VAD session.
    pub fn session_time(&self) -> Duration {
        Duration::from_secs_f64(self.processed_samples as f64 / self.config.sample_rate as f64)
    }

    /// Reset the status of the model
    // TODO should this reset the audio buffer as well?
    pub fn reset(&mut self) {
        self.state_tensor = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        self.speech_start = None;
        self.speech_end = None;
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
        Duration::from_millis((self.silent_samples / (self.config.sample_rate / 1000)) as u64)
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            // https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
            positive_speech_threshold: 0.5,
            negative_speech_threshold: 0.35,
            pre_speech_pad: Duration::from_millis(600),
            redemption_time: Duration::from_millis(600),
            sample_rate: 16000,
            min_speech_time: Duration::from_millis(90),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic smoke test that the model loads correctly and we haven't committed rubbish to the
    /// repo.
    #[test]
    fn model_loads() {
        let _sesion = VadSession::new(VadConfig::default()).unwrap();
        let _sesion =
            VadSession::new_from_path("models/silero_vad.onnx", VadConfig::default()).unwrap();
    }

    /// Too short tensors result in inference errors which we don't want to unnecessarily bubble up
    /// to the user and instead handle in our buffering implementation. This test will check that a
    /// short inference in the internal inference call bubbles up an error but when using the
    /// public API no error is presented.
    #[test]
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
    fn silence_handling() {
        let mut session = VadSession::new(VadConfig::default()).unwrap();
        let silence = vec![0.0; 30 * 16]; // 30ms of silence

        assert!(session.process(&silence).unwrap().is_empty());
        assert_eq!(session.processed_samples, silence.len());
    }

    /// We only allow for 8khz and 16khz audio.
    #[test]
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
}

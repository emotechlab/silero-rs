#![doc = include_str!("../README.md")]
use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::{GraphOptimizationLevel, Session};
use std::ops::Range;
use std::path::Path;

/// Parameters used to configure a vad session. These will determine the sensitivity and switching
/// speed of detection.
#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub pre_speech_pad_ms: usize,
    pub redemption_frames: usize,
    pub sample_rate: usize,
    pub min_speech_frames: usize,
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
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
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
        min_frames_passed: bool,
        speech_frames: usize,
        redemption_frames: usize,
    },
    Silence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        self.session_audio.extend_from_slice(audio_frame);

        let vad_segment_length = VAD_BUFFER_MS * self.config.sample_rate / 1000;
        let num_chunks = self.session_audio.len() / vad_segment_length;
        let start_chunk = self.processed_samples / vad_segment_length;

        let mut transitions = vec![];

        for i in start_chunk..num_chunks {
            let start_idx = i * vad_segment_length;
            // we might not be getting audio chunks in perfect multiples of 30ms, so let the
            // last frame accommodate the remainder. This adds a bit of non-determinism based on
            // audio size but it does let us more eagerly process audio.
            let sample_range = if i < num_chunks - 1 {
                start_idx..(start_idx + vad_segment_length)
            } else {
                start_idx..self.session_audio.len()
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
        let audio_tensor = Array2::from_shape_vec((1, samples), audio_frame.to_vec())?;
        let result = self.model.run(ort::inputs![
            audio_tensor.view(),
            self.sample_rate_tensor.view(),
            self.h_tensor.view(),
            self.c_tensor.view()
        ]?)?;

        // Update internal state tensors.
        self.h_tensor = result
            .get("hn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .to_owned()
            .into_shape((2, 1, 64))
            .expect("Shape mismatch for h_tensor");
        self.c_tensor = result
            .get("cn")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .to_owned()
            .into_shape((2, 1, 64))
            .expect("Shape mismatch for h_tensor");

        let prob = *result
            .get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .first()
            .unwrap();

        let mut vad_change = None;

        if prob < self.config.negative_speech_threshold {
            self.silent_samples += samples;
        } else {
            self.silent_samples = 0;
        }

        match self.state {
            VadState::Silence => {
                if prob > self.config.positive_speech_threshold {
                    self.state = VadState::Speech {
                        start_ms: self
                            .session_time()
                            .saturating_sub(self.config.pre_speech_pad_ms),
                        min_frames_passed: false,
                        speech_frames: 0,
                        redemption_frames: 0,
                    };
                }
            }
            VadState::Speech {
                start_ms,
                ref mut min_frames_passed,
                ref mut speech_frames,
                ref mut redemption_frames,
            } => {
                *speech_frames += 1;
                if !*min_frames_passed && *speech_frames > self.config.min_speech_frames {
                    *min_frames_passed = true;
                    // TODO: the pre speech padding should not cross over the previous speech->silence
                    // transition, if there was one
                    vad_change = Some(VadTransition::SpeechStart {
                        timestamp_ms: start_ms,
                    });
                    self.speech_start = Some(start_ms);
                    self.speech_end = None;
                }

                if prob < self.config.negative_speech_threshold {
                    if !*min_frames_passed {
                        self.state = VadState::Silence;
                    } else {
                        *redemption_frames += 1;
                        if *redemption_frames > self.config.redemption_frames {
                            if *min_frames_passed {
                                let speech_end = (self.processed_samples + audio_frame.len())
                                    / (self.config.sample_rate / 1000);
                                vad_change = Some(VadTransition::SpeechEnd {
                                    timestamp_ms: speech_end,
                                });
                                self.speech_end = Some(speech_end);
                            }
                            self.state = VadState::Silence
                        }
                    }
                } else {
                    *redemption_frames = 0;
                }
            }
        };

        self.processed_samples += audio_frame.len();

        Ok(vad_change)
    }

    /// Returns whether the vad current believes the audio to contain speech
    pub fn is_speaking(&self) -> bool {
        matches!(self.state, VadState::Speech {
            min_frames_passed, ..
        } if min_frames_passed)
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
    pub fn current_speech_len(&self) -> usize {
        self.get_current_speech().len()
    }

    /// Get the current length of the VAD session.
    pub fn session_time(&self) -> usize {
        self.processed_samples / (self.config.sample_rate / 1000)
    }

    /// Reset the status of the model
    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.speech_start = None;
        self.speech_end = None;
        self.state = VadState::Silence;
    }

    /// Returns the length of the end silence. The VAD may be showing this as speaking because of
    /// redemption frames or other parameters that slow down the speed it can switch at. But this
    /// measure is a raw unprocessed look of how many segments since the last speech are below the
    /// negative speech threshold.
    pub fn end_silence_length(&self) -> usize {
        self.silent_samples
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            // https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
            positive_speech_threshold: 0.35,
            negative_speech_threshold: 0.25,
            pre_speech_pad_ms: 600,
            redemption_frames: 20,
            sample_rate: 16000,
            min_speech_frames: 3,
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

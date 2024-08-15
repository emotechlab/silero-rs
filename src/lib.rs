use anyhow::{bail, Result};
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::{GraphOptimizationLevel, Session};

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
    pub fn get_frame_samples(&self) -> usize {
        (30_f32 / 1000_f32 * self.sample_rate as f32) as usize // 30ms * sample_rate Hz
    }

    pub fn get_frames(length_ms: usize) -> usize {
        length_ms / 30
    }
}

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
    speech_start: Option<usize>,
    speech_end: Option<usize>,
}

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
    pub fn new(config: VadConfig) -> Result<Self> {
        if ![8000_usize, 16000].contains(&config.sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }
        let model_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/silero_vad.onnx"
        ));
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
            speech_start: None,
            speech_end: None,
        })
    }

    /// Advance the VAD state machine with an audio frame. Keep between 30-96ms in length.
    /// Return indicates if a transition from speech to silence (or silence to speech) occurred.
    ///
    /// Important: don't implement your own endpointing logic.
    /// Instead, when a `SpeechEnd` is returned, you can use the `get_current_speech()` method to retrieve the audio.
    pub fn process(&mut self, audio_frame: &[f32]) -> Result<Option<VadTransition>> {
        self.session_audio.extend_from_slice(audio_frame);

        let audio_tensor = Array2::from_shape_vec((1, audio_frame.len()), audio_frame.to_vec())?;
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

    pub fn is_speaking(&self) -> bool {
        matches!(self.state, VadState::Speech {
            min_frames_passed, ..
        } if min_frames_passed)
    }

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

    pub fn current_speech_len(&self) -> usize {
        self.get_current_speech().len()
    }

    pub fn session_time(&self) -> usize {
        self.processed_samples / (self.config.sample_rate / 1000)
    }

    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.speech_start = None;
        self.speech_end = None;
        self.state = VadState::Silence;
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

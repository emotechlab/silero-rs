use hound::WavReader;
use silero::*;
use std::fmt;
use std::path::Path;

#[test]
fn compare_audio_1() {
    let audio = Path::new("tests/audio/sample_1.wav");
    compare_audio(&audio);
}

#[test]
fn compare_audio_2() {
    let audio = Path::new("tests/audio/sample_2.wav");
    compare_audio(&audio);
}

#[test]
fn compare_audio_3() {
    let audio = Path::new("tests/audio/sample_3.wav");
    compare_audio(&audio);
}

#[test]
fn compare_audio_4() {
    let audio = Path::new("tests/audio/sample_4.wav");
    compare_audio(&audio);
}

#[test]
fn compare_birds() {
    let audio = Path::new("tests/audio/birds.wav");
    compare_audio(&audio);
}

fn compare_audio(audio: &Path) {
    let config = VadConfig::default();

    let whole_file = silero_whole_file(audio, config.clone());

    let chunks_30ms = silero_streaming(audio, 30, config.clone());
    assert_eq!(whole_file, chunks_30ms);
}

#[derive(Clone, PartialEq, Eq)]
struct Segment {
    start_timestamp_ms: usize,
    end_timestamp_ms: usize,
    /// Convert to i16 to make comparing samples easier
    samples: Vec<i16>,
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("start_time_ms", &self.start_timestamp_ms)
            .field("end_time_ms", &self.end_timestamp_ms)
            .field("samples::len", &self.samples.len())
            .finish()
    }
}

fn silero_whole_file(audio: &Path, config: VadConfig) -> Vec<Segment> {
    let step = if config.sample_rate == 16000 {
        1
    } else {
        2 // Other sample rates are invalid so we'll just work on less data
    };
    let samples: Vec<f32> = WavReader::open(audio)
        .unwrap()
        .into_samples()
        .step_by(step)
        .map(|x| {
            let modified = x.unwrap_or(0i16) as f32 / (i16::MAX as f32);
            modified.clamp(-1.0, 1.0)
        })
        .collect();

    let len = samples.len();
    inner_vad_process(samples, len, config)
}

fn silero_streaming(audio: &Path, chunk_ms: usize, config: VadConfig) -> Vec<Segment> {
    let chunk_size = (config.sample_rate * chunk_ms) / 1000;
    let step = if config.sample_rate == 16000 {
        1
    } else {
        2 // Other sample rates are invalid so we'll just work on less data
    };
    let samples: Vec<f32> = WavReader::open(audio)
        .unwrap()
        .into_samples()
        .step_by(step)
        .map(|x| {
            let modified = x.unwrap_or(0i16) as f32 / (i16::MAX as f32);
            modified.clamp(-1.0, 1.0)
        })
        .collect();

    inner_vad_process(samples, chunk_size, config)
}

fn inner_vad_process(samples: Vec<f32>, chunk_size: usize, config: VadConfig) -> Vec<Segment> {
    let mut result = vec![];
    let mut session = VadSession::new(config.clone()).unwrap();
    let num_chunks = samples.len() / chunk_size;
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = if i < num_chunks - 1 {
            start + chunk_size
        } else {
            samples.len()
        };

        let mut transitions = session.process(&samples[start..end]).unwrap();
        for transition in transitions.drain(..) {
            if let VadTransition::SpeechEnd {
                start_timestamp_ms,
                end_timestamp_ms,
                samples,
            } = transition
            {
                let samples = samples
                    .iter()
                    .map(|x| (*x * i16::MAX as f32) as i16)
                    .collect();
                result.push(Segment {
                    start_timestamp_ms,
                    end_timestamp_ms,
                    samples,
                });
            }
        }
    }

    if session.is_speaking() {
        let end = session.session_time();
        let start = end - session.current_speech_duration();

        let samples = session
            .get_current_speech()
            .iter()
            .map(|x| (*x * i16::MAX as f32) as i16)
            .collect();
        result.push(Segment {
            start_timestamp_ms: start.as_millis() as usize,
            end_timestamp_ms: end.as_millis() as usize,
            samples,
        });
    }

    result
}

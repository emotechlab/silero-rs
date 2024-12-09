use hound::WavReader;
use silero::*;
use std::fmt;
use std::path::Path;
use tracing_test::traced_test;

#[test]
#[traced_test]
fn compare_audio_1() {
    let audio = Path::new("tests/audio/sample_1.wav");
    compare_audio(&audio);
}

#[test]
#[traced_test]
fn compare_audio_2() {
    let audio = Path::new("tests/audio/sample_2.wav");
    compare_audio(&audio);
}

#[test]
#[traced_test]
fn compare_audio_3() {
    let audio = Path::new("tests/audio/sample_3.wav");
    compare_audio(&audio);
}

#[test]
#[traced_test]
fn compare_audio_4() {
    let audio = Path::new("tests/audio/sample_4.wav");
    compare_audio(&audio);
}

#[test]
#[traced_test]
fn compare_birds() {
    let audio = Path::new("tests/audio/birds.wav");
    compare_audio(&audio);
}

#[test]
#[traced_test]
fn include_unprocessed_audio_1() {
    let audio = Path::new("tests/audio/sample_1.wav");
    // Use prime number as chunk duration so we are more confident that the code indeed works. We also test common
    // practical choices: 20, 30, 50.
    for chunk_ms in [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 20, 30, 50,
    ] {
        should_include_unprocessed_when_is_speaking(audio, chunk_ms);
    }
}

#[test]
#[traced_test]
fn include_unprocessed_audio_2() {
    let audio = Path::new("tests/audio/sample_2.wav");
    for chunk_ms in [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 20, 30, 50,
    ] {
        should_include_unprocessed_when_is_speaking(audio, chunk_ms);
    }
}

#[test]
#[traced_test]
fn include_unprocessed_audio_3() {
    let audio = Path::new("tests/audio/sample_3.wav");
    for chunk_ms in [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 20, 30, 50,
    ] {
        should_include_unprocessed_when_is_speaking(audio, chunk_ms);
    }
}

#[test]
#[traced_test]
fn include_unprocessed_audio_4() {
    let audio = Path::new("tests/audio/sample_4.wav");
    for chunk_ms in [20, 30, 50] {
        should_include_unprocessed_when_is_speaking(audio, chunk_ms);
    }
}

/// When is_speaking is true, get_current_speech should return everything starting from
/// speech start time until the very end, including the unprocessed samples.
fn should_include_unprocessed_when_is_speaking(audio: &Path, chunk_ms: usize) {
    let mut test_executed = false;

    let config = VadConfig::default();
    let mut vad = VadSession::new(config).unwrap();

    // Read audio samples.
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

    // Send each chunk to vad for inference.
    let chunk_size = (config.sample_rate * chunk_ms) / 1000;
    let num_chunks = samples.len() / chunk_size;
    let mut last_speech_start_ms = 0;
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = if i < num_chunks - 1 {
            start + chunk_size
        } else {
            samples.len()
        };

        let transitions = vad.process(&samples[start..end]).unwrap();
        // Update last_speech_start_ms
        for transition in transitions {
            if let VadTransition::SpeechStart { timestamp_ms } = transition {
                last_speech_start_ms = timestamp_ms;
            }
        }

        // The actual test logic.
        if vad.is_speaking() {
            test_executed = true;
            let current_speech = vad.get_current_speech();
            let total_time_send_to_vad = sample_nums_to_ms(end, &config);
            let current_speech_end_time =
                last_speech_start_ms + sample_nums_to_ms(current_speech.len(), &config);
            assert_eq!(total_time_send_to_vad, current_speech_end_time);
        }
    }
    assert!(test_executed);
}

#[inline]
fn sample_nums_to_ms(sample_num: usize, config: &VadConfig) -> usize {
    ((1000 * sample_num) as f32 / config.sample_rate as f32) as usize
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

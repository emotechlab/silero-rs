use hound::WavReader;
use silero::*;
use std::fmt;
use std::path::Path;
use tracing_test::traced_test;

#[derive(Copy, Clone, Debug)]
enum ChunkStrategy {
    Fixed(usize),
    Varying((usize, usize)),
}

impl ChunkStrategy {
    fn get_chunk_size(&self, config: &VadConfig) -> usize {
        match self {
            Self::Fixed(size) => (config.sample_rate * size) / 1000,
            Self::Varying((min, max)) => (config.sample_rate * fastrand::usize(min..max)) / 1000,
        }
    }
}

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

fn compare_audio(audio: &Path) {
    let config = VadConfig::default();

    let whole_file = silero_whole_file(audio, config.clone());

    let chunks_30ms = silero_streaming(audio, ChunkStrategy::Fixed(30), config.clone());
    assert_eq!(whole_file, chunks_30ms);

    // 20ms initial remainder
    let chunks_100ms = silero_streaming(audio, ChunkStrategy::Fixed(100), config.clone());
    assert_eq!(whole_file, chunks_100ms);

    // 1ms initial remainder
    let chunks_31ms = silero_streaming(audio, ChunkStrategy::Fixed(31), config.clone());
    assert_eq!(whole_file, chunks_31ms);

    // vary from 5-500ms chunks
    let chunks_random = silero_streaming(audio, ChunkStrategy::Varying((5, 500)), config.clone());
    assert_eq!(whole_file, chunks_random);
    panic!();
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
    inner_vad_process(samples, ChunkStrategy::Fixed(len), config)
}

fn silero_streaming(audio: &Path, chunks: ChunkStrategy, config: VadConfig) -> Vec<Segment> {
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

    inner_vad_process(samples, chunks, config)
}

fn inner_vad_process(samples: Vec<f32>, chunks: ChunkStrategy, config: VadConfig) -> Vec<Segment> {
    let mut result = vec![];
    let mut session = VadSession::new(config.clone()).unwrap();

    let mut start = 0;
    let mut end = 0;

    while end < samples.len() {
        let chunk_size = chunks.get_chunk_size(&config);
        end = samples.len().min(start + chunk_size);

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
        start = end;
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

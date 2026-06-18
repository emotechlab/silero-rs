use divan::{black_box, Bencher};
use hound::WavReader;
use ndarray::{Array1, Array2, Array3, ArrayBase, Ix1, Ix3, OwnedRepr};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use silero::VadSession;
use std::time::Duration;

const MODEL_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/silero_vad.onnx"
));

fn main() {
    divan::main();
}

#[divan::bench]
fn startup_rten() -> VadSession {
    VadSession::new(Default::default()).expect("RTen VAD session should load")
}

#[divan::bench]
fn startup_ort() -> Session {
    ort_session_from_bytes()
}

#[divan::bench(args = [8000, 16000])]
fn chunk_inference_rten(bencher: Bencher, sample_rate: usize) {
    bencher
        .with_inputs(|| {
            let mut config = silero::VadConfig::default();
            config.sample_rate = sample_rate;
            let session = VadSession::new(config).expect("RTen VAD session should load");
            let frame = vec![0.0; sample_rate * 30 / 1000];
            (session, frame)
        })
        .bench_local_refs(|(session, frame)| {
            session
                .speech_probability(black_box(frame))
                .expect("RTen inference should succeed")
        });
}

#[divan::bench(args = [8000, 16000])]
fn chunk_inference_ort(bencher: Bencher, sample_rate: usize) {
    bencher
        .with_inputs(|| {
            let session = OrtVadSession::new(sample_rate);
            let frame = vec![0.0; sample_rate * 30 / 1000];
            (session, frame)
        })
        .bench_local_refs(|(session, frame)| session.forward(black_box(frame)));
}

#[divan::bench(args = [20, 30, 50, 100])]
fn process_file(chunk_ms: usize) {
    let chunk_size = chunk_ms * 16; // 16000/1000

    let mut session = VadSession::new(Default::default()).expect("RTen VAD session should load");
    let samples = read_wav("tests/audio/sample_2.wav");

    let num_chunks = samples.len() / chunk_size;
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = if i < num_chunks - 1 {
            start + chunk_size
        } else {
            samples.len()
        };

        let _transitions = session
            .process(&samples[start..end])
            .expect("RTen VAD processing should succeed");
    }
}

#[divan::bench(args = [1, 2, 3, 4, 5, 6])]
fn take(to_take: u64) {
    let mut session = VadSession::new(Default::default()).expect("RTen VAD session should load");
    let samples = read_wav("tests/audio/rooster.wav");

    let _ = session.process(&samples);

    let _ = session.take_until(Duration::from_secs(to_take));
}

#[divan::bench]
fn push_multiple_silences() {
    let mut session = VadSession::new(Default::default()).expect("RTen VAD session should load");
    let silence = vec![0.0; 500];
    // Push 500ms of silence
    for _ in 0..16 {
        let _ = session.process(&silence);
    }
}

fn read_wav(path: &str) -> Vec<f32> {
    WavReader::open(path)
        .expect("benchmark WAV should open")
        .into_samples::<i16>()
        .map(|sample| {
            let modified =
                sample.expect("benchmark WAV sample should decode") as f32 / (i16::MAX as f32);
            modified.clamp(-1.0, 1.0)
        })
        .collect()
}

struct OrtVadSession {
    model: Session,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate_tensor: ArrayBase<OwnedRepr<i64>, Ix1>,
}

impl OrtVadSession {
    fn new(sample_rate: usize) -> Self {
        Self {
            model: ort_session_from_bytes(),
            h_tensor: Array3::<f32>::zeros((2, 1, 64)),
            c_tensor: Array3::<f32>::zeros((2, 1, 64)),
            sample_rate_tensor: Array1::from_vec(vec![sample_rate as i64]),
        }
    }

    fn forward(&mut self, input: &[f32]) -> f32 {
        let audio_tensor = Array2::from_shape_vec((1, input.len()), input.to_vec())
            .expect("benchmark frame should match tensor shape");
        let mut result = self
            .model
            .run(ort::inputs![
                TensorRef::from_array_view(audio_tensor.view())
                    .expect("audio tensor should convert to ORT input"),
                TensorRef::from_array_view(self.sample_rate_tensor.view())
                    .expect("sample-rate tensor should convert to ORT input"),
                TensorRef::from_array_view(self.h_tensor.view())
                    .expect("h tensor should convert to ORT input"),
                TensorRef::from_array_view(self.c_tensor.view())
                    .expect("c tensor should convert to ORT input")
            ])
            .expect("ORT inference should succeed");

        self.h_tensor = result
            .get("hn")
            .expect("ORT output should contain hn")
            .try_extract_array::<f32>()
            .expect("hn should be an f32 tensor")
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .expect("hn should have recurrent state shape");

        self.c_tensor = result
            .get("cn")
            .expect("ORT output should contain cn")
            .try_extract_array::<f32>()
            .expect("cn should be an f32 tensor")
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .expect("cn should have recurrent state shape");

        let output = result
            .remove("output")
            .expect("ORT output should contain probability");
        *output
            .try_extract_array::<f32>()
            .expect("output should be an f32 tensor")
            .first()
            .expect("output should contain a probability")
    }
}

fn ort_session_from_bytes() -> Session {
    Session::builder()
        .expect("ORT session builder should initialize")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("ORT graph optimization should be configurable")
        .with_intra_threads(4)
        .expect("ORT intra-op thread count should be configurable")
        .commit_from_memory(MODEL_BYTES)
        .expect("ORT should load the Silero ONNX model")
}

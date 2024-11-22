use hound::WavReader;
use silero::VadSession;

fn main() {
    divan::main();
}

#[divan::bench(args = [20, 30, 50, 100])]
fn process_file(chunk_ms: usize) {
    let chunk_size = chunk_ms * 16; // 16000/1000

    let mut session = VadSession::new(Default::default()).unwrap();
    let samples: Vec<f32> = WavReader::open("tests/audio/sample_2.wav")
        .unwrap()
        .into_samples()
        .map(|x| {
            let modified = x.unwrap_or(0i16) as f32 / (i16::MAX as f32);
            modified.clamp(-1.0, 1.0)
        })
        .collect();

    let num_chunks = samples.len() / chunk_size;
    let mut last_end = 0;
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = if i < num_chunks - 1 {
            start + chunk_size
        } else {
            samples.len()
        };

        let _transitions = session.process(&samples[start..end]).unwrap();
    }
}

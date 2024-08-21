//! Snapshot tests of the VAD behaviour.
//!
//! The general approach of this is snapshot testing like we might see with a crate like
//! [insta](crates.io/crates/insta) but we're not going to use insta? Why because it is a bit
//! fiddly to fit it into a workflow where test failures may need an analysis of the data and
//! neural network output to see if the snapshot should be updated. And without being able to make
//! use of the diff-review and accepting tools it seems easier to roll our own solution specialised
//! to the application.
//!
//! So here I'll make a report type, and for each test we'll compare to an existing report and also
//! save the new reports if they differ. And provide some scripts to generally plot the data in the
//! reports versus the audio and help us debug via charts! Each test will be ran for all audio in
//! the audio folder as well.
//!
//! Also all test audios will be 16kHz to make it easy to test silero in both 16kHz and 8kHz modes.
use hound::WavReader;
use serde::{Deserialize, Serialize};
use silero::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct Report {
    transitions: Vec<VadTransition>,
    silence_samples: Vec<usize>,
}

fn get_audios() -> Vec<PathBuf> {
    let audio_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/audio"));
    let mut result = vec![];
    for entry in fs::read_dir(&audio_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.is_dir() {
            result.push(path.to_path_buf());
        }
    }
    result
}

fn get_chunk_size(sample_rate: usize, chunk_ms: usize) -> usize {
    (sample_rate * chunk_ms) / 1000
}

#[test]
fn chunk_30_default_params_16k() {
    let audios = get_audios();

    let config = VadConfig::default();

    let chunk_size = get_chunk_size(16000, 30);

    let mut summary = BTreeMap::new();

    for audio in audios.iter() {
        let mut report = Report::default();
        let mut session = VadSession::new(config.clone()).unwrap();
        let samples: Vec<f32> = WavReader::open(&audio)
            .unwrap()
            .into_samples()
            .map(|x| x.unwrap_or(0i16) as f32)
            .collect();

        let num_chunks = samples.len() / chunk_size;

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = if i < num_chunks - 1 {
                start + chunk_size
            } else {
                samples.len()
            };

            let mut transitions = session.process(&samples[start..end]).unwrap();
            report.transitions.append(&mut transitions);
            report
                .silence_samples
                .push(session.current_silence_samples());
        }
        summary.insert(audio.to_path_buf(), report);
    }

    let report = serde_json::to_string_pretty(&summary).unwrap();

    let name = "chunk_30_default_params_16k.json";

    fs::write(Path::new("target").join(name), &report).unwrap();

    let expected = fs::read_to_string(
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data")).join(name),
    )
    .unwrap();
    let expected: BTreeMap<PathBuf, Report> = serde_json::from_str(&expected).unwrap();

    assert_eq!(expected, summary);
}

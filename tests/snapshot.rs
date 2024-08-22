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

#[derive(Default, Debug, PartialEq, Deserialize, Serialize)]
struct Summary {
    config: VadConfig,
    summary: BTreeMap<PathBuf, Report>,
}

#[derive(Default, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct Report {
    transitions: Vec<VadTransition>,
    current_silence_samples: Vec<usize>,
    current_speech_samples: Vec<usize>,
}

#[test]
fn chunk_50_default_params_16k() {
    run_snapshot_test(50, VadConfig::default(), "default");
}

#[test]
#[ignore]
fn chunk_50_default_params_8k() {
    let mut config = VadConfig::default();
    config.sample_rate = 8000;
    run_snapshot_test(50, config, "default");
}

#[test]
fn chunk_30_default_params_16k() {
    run_snapshot_test(30, VadConfig::default(), "default");
}

#[test]
fn chunk_30_default_params_8k() {
    let mut config = VadConfig::default();
    config.sample_rate = 8000;
    run_snapshot_test(30, config, "default");
}

fn run_snapshot_test(chunk_ms: usize, config: VadConfig, config_name: &str) {
    let audios = get_audios();

    let chunk_size = get_chunk_size(config.sample_rate, chunk_ms);

    let mut summary = BTreeMap::new();

    for audio in audios.iter() {
        let mut session = VadSession::new(config.clone()).unwrap();
        let mut report = Report::default();
        let step = if config.sample_rate == 16000 {
            1
        } else {
            2 // Other sample rates are invalid so we'll just work on less data
        };
        let samples: Vec<f32> = WavReader::open(&audio)
            .unwrap()
            .into_samples()
            .step_by(step)
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
                .current_silence_samples
                .push(session.current_silence_samples());
            report
                .current_speech_samples
                .push(session.current_speech_samples());
        }
        summary.insert(audio.to_path_buf(), report);
    }

    let summary = Summary { summary, config };
    let report = serde_json::to_string_pretty(&summary).unwrap();

    let name = format!(
        "chunk_{}_{}Hz_{}.json",
        chunk_ms, config.sample_rate, config_name
    );

    let current_report = Path::new("target").join(&name);
    let baseline_report = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data")).join(&name);

    fs::write(&current_report, &report).unwrap();

    println!("Loading snapshot from: {}", baseline_report.display());
    let expected = fs::read_to_string(&baseline_report).unwrap();
    let expected: Summary = serde_json::from_str(&expected).unwrap();

    println!(
        "Checking current results {} with snapshot {}",
        current_report.display(),
        baseline_report.display()
    );

    // TODO here we want to be a bit nicer and iterate over the files and either:
    // 1. Iterate over the files and compare each one and generate python script commands to plot
    //    and inspect
    // 2. Have our python script be able to take two reports and generate plots that only concern
    //    the deltas!
    assert_eq!(expected, summary);
}

fn get_audios() -> Vec<PathBuf> {
    let audio_dir = Path::new("tests/audio");
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

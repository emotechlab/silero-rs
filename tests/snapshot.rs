//! Snapshot tests of the VAD behaviour.
//!
//! The general approach of this is snapshot testing like we might see with a crate like
//! [insta](crates.io/crates/insta) but we're not going to use insta. Why? Because it is a bit
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
use approx::assert_ulps_eq;
use hound::WavReader;
use serde::{Deserialize, Serialize};
use silero::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default, Debug, PartialEq, Deserialize, Serialize)]
struct Summary {
    input_size_ms: usize,
    config: VadConfig,
    summary: BTreeMap<PathBuf, Report>,
}

#[derive(Default, Debug, PartialEq, Deserialize, Serialize)]
struct Report {
    transitions: Vec<VadTransition>,
    current_silence_samples: Vec<usize>,
    current_speech_samples: Vec<usize>,
    likelihoods: Vec<usize>,
}

#[test]
fn chunk_50_default_params_16k() {
    run_snapshot_test(50, VadConfig::default(), "default");
}

#[test]
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

#[test]
fn chunk_20_default_params_16k() {
    run_snapshot_test(20, VadConfig::default(), "default");
}

#[test]
fn chunk_20_default_params_8k() {
    let mut config = VadConfig::default();
    config.sample_rate = 8000;
    run_snapshot_test(20, config, "default");
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
            .map(|x| x.unwrap_or(0i16) as f32 / (i16::MAX as f32))
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

            let mut transitions = session.process(&samples[start..end]).unwrap();
            report.transitions.append(&mut transitions);
            report
                .current_silence_samples
                .push(session.current_silence_samples());
            report
                .current_speech_samples
                .push(session.current_speech_samples());

            if let Ok(network_outputs) = session.forward(samples[last_end..end].to_vec()) {
                let prob = *network_outputs
                    .try_extract_tensor::<f32>()
                    .unwrap()
                    .first()
                    .unwrap()
                    * 100.0;
                report.likelihoods.push(prob as usize);
                // Try and solve the too small inference issue
                last_end = end;
            }
        }
        summary.insert(audio.to_path_buf(), report);
    }

    let summary = Summary {
        input_size_ms: chunk_ms,
        summary,
        config,
    };
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

    // Lets do some basic checks first just to make sure we're not complete trash
    println!("Checking snapshot is generated with same configuration!");
    compare_configs(&summary.config, &expected.config);
    assert_eq!(summary.input_size_ms, expected.input_size_ms);

    let mut failing_files = vec![];
    println!();

    for sample in summary.summary.keys() {
        let baseline = &summary.summary[sample];
        let current = &expected.summary[sample];

        if baseline != current {
            println!("{} is failing", sample.display());
            if baseline.transitions != current.transitions {
                println!("\tDifference in transitons list!");
            }
            if baseline.current_silence_samples != current.current_silence_samples {
                println!("\tDifference in silence lengths");
            }
            if baseline.current_speech_samples != current.current_speech_samples {
                println!("\tDifference in speech lengths");
            }
            failing_files.push(sample.to_path_buf());
        }
    }
    if !failing_files.is_empty() {
        println!();
        println!("You have some failing files and targets. If you get a snapshot file and audio you can plot it via our plot_audio script e.g.");
        println!();
        println!(
            "python3 scripts/plot_audio.py -a {} -i {}",
            failing_files[0].display(),
            current_report.display()
        );
        println!();

        panic!("The following files are failing: {:?}", failing_files);
    }
}

fn compare_configs(a: &VadConfig, b: &VadConfig) {
    assert_ulps_eq!(a.positive_speech_threshold, b.positive_speech_threshold);
    assert_ulps_eq!(a.negative_speech_threshold, b.negative_speech_threshold);
    assert_eq!(a.pre_speech_pad, b.pre_speech_pad);
    assert_eq!(a.redemption_time, b.redemption_time);
    assert_eq!(a.sample_rate, b.sample_rate);
    assert_eq!(a.min_speech_time, b.min_speech_time);
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

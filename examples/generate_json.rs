use clap::Parser;
use hound::WavReader;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use silero::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
struct Summary {
    input_size_ms: usize,
    config: VadConfig,
    summary: BTreeMap<PathBuf, Report>,
}

#[derive(Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
struct Report {
    transitions: Vec<VadTransition>,
    current_session_samples: Vec<usize>,
    current_silence_samples: Vec<usize>,
    current_speech_samples: Vec<usize>,
    likelihoods: Vec<usize>,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, short)]
    input: PathBuf,
    #[arg(long, short, default_value = "output.json")]
    output: PathBuf,
    #[arg(long, default_value = "0.5")]
    positive_speech_threshold: f32,
    #[arg(long, default_value = "0.35")]
    negative_speech_threshold: f32,
    #[arg(long, default_value = "600")]
    redemption_time_ms: u64,
    #[arg(long, default_value = "600")]
    pre_speech_pad_ms: u64,
    #[arg(long, default_value = "90")]
    min_speech_time_ms: u64,
    #[arg(long, default_value = "0")]
    post_speech_pad_ms: u64,
}

fn main() {
    let args = Args::parse();
    run_snapshot_test(50, args);
}

fn get_chunk_size(sample_rate: usize, chunk_ms: usize) -> usize {
    (sample_rate * chunk_ms) / 1000
}

#[cfg(not(feature = "serde"))]
fn run_snapshot_test(_chunk_ms: usize, _args: Args) {
    panic!("Can't run snapshot test serde feature not enabled. To run add `--all-features` or `--feature serde` to cargo invocation");
}

#[cfg(feature = "serde")]
fn run_snapshot_test(chunk_ms: usize, args: Args) {
    let mut config = VadConfig {
        positive_speech_threshold: args.positive_speech_threshold,
        negative_speech_threshold: args.negative_speech_threshold,
        pre_speech_pad: Duration::from_millis(args.pre_speech_pad_ms),
        post_speech_pad: Duration::from_millis(args.post_speech_pad_ms),
        redemption_time: Duration::from_millis(args.redemption_time_ms),
        min_speech_time: Duration::from_millis(args.min_speech_time_ms),
        ..Default::default()
    };

    let mut summary = BTreeMap::new();

    let audio = &args.input;
    let mut session = VadSession::new(config.clone()).unwrap();
    let mut report = Report::default();
    let reader = WavReader::open(&audio).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "only supports mono audio");
    let samples: Vec<f32> = reader
        .into_samples()
        .map(|x| {
            let modified = x.unwrap_or(0i16) as f32 / (i16::MAX as f32);
            modified.clamp(-1.0, 1.0)
        })
        .collect();

    config.sample_rate = spec.sample_rate as usize;
    let chunk_size = get_chunk_size(config.sample_rate, chunk_ms);

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
        report
            .current_session_samples
            .push(session.session_audio_samples());

        if let Ok(network_outputs) = session.forward(samples[last_end..end].to_vec()) {
            let prob = *network_outputs
                .try_extract_array::<f32>()
                .unwrap()
                .first()
                .unwrap()
                * 100.0;
            report.likelihoods.push(prob as usize);
            // Try and solve the too small inference issue
            last_end = end;
        }
    }
    summary.insert(PathBuf::from(audio), report);

    let summary = Summary {
        input_size_ms: chunk_ms,
        summary,
        config,
    };
    let report = serde_json::to_string_pretty(&summary).unwrap();

    fs::write(&args.output, report).unwrap();
}

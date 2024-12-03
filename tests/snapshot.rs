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
//! Also, all test audios will be 16kHz to make it easy to test silero in both 16kHz and 8kHz modes.
use approx::assert_ulps_eq;
use hound::WavReader;
use serde::{Deserialize, Serialize};
use silero::*;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing_test::traced_test;

#[derive(Default, Debug, Deserialize, Serialize)]
struct Summary {
    input_size_ms: usize,
    config: VadConfig,
    summary: BTreeMap<PathBuf, Report>,
}

impl Summary {
    /// Different config leads to different summary. This eq method will consider this point when
    /// checking equality by calculating the expected difference introduced by config modifications.
    ///
    /// This method uses None to represent no difference, and Some(HashMap<PathBuf, Vec<String>>) to
    /// report all the differences found in each file.
    fn eq(&self, other: &Self) -> Option<HashMap<PathBuf, Vec<String>>> {
        // TODO: what if input_size_ms is different?
        let mut errors: HashMap<PathBuf, Vec<String>> = HashMap::new();

        assert_eq!(self.summary.len(), other.summary.len());
        for (audio_file, baseline) in &self.summary {
            match other.summary.get(audio_file) {
                None => {
                    errors.entry(audio_file.clone()).or_default().push(format!(
                        "Cannot find report for audio file {}",
                        audio_file.display()
                    ));
                }
                Some(current) => {
                    if !baseline.eq_transitions(current, &self.config, &other.config) {
                        errors
                            .entry(audio_file.clone())
                            .or_default()
                            .push("Difference in transitions".to_string());
                    }
                    if !baseline.eq_current_silence_samples(current, &self.config, &other.config) {
                        errors
                            .entry(audio_file.clone())
                            .or_default()
                            .push("Difference in current_silence_samples".to_string());
                    }
                    if !baseline.eq_current_speech_samples(current, &self.config, &other.config) {
                        errors
                            .entry(audio_file.clone())
                            .or_default()
                            .push("Difference in speech_samples".to_string());
                    }
                }
            }
        }

        if errors.is_empty() {
            None
        } else {
            Some(errors)
        }
    }
}

#[derive(Default, Debug, PartialEq, Deserialize, Serialize)]
struct Report {
    transitions: Vec<VadTransition>,
    current_session_samples: Vec<usize>,
    current_silence_samples: Vec<usize>,
    current_speech_samples: Vec<usize>,
    likelihoods: Vec<usize>,
}

impl Report {
    fn eq_transitions(
        &self,
        other: &Self,
        self_config: &VadConfig,
        other_config: &VadConfig,
    ) -> bool {
        // Some differences are introduced by modified config and they are expected.
        let allowed_post_pad_diff = (self_config.post_speech_pad.as_millis() as isize
            - other_config.post_speech_pad.as_millis() as isize)
            .abs() as usize;

        for (baseline, current) in self.transitions.iter().zip(other.transitions.iter()) {
            match (baseline, current) {
                (
                    VadTransition::SpeechStart { timestamp_ms: ts_1 },
                    VadTransition::SpeechStart { timestamp_ms: ts_2 },
                ) => {
                    if ts_1 != ts_2 {
                        return false;
                    }
                }
                (
                    VadTransition::SpeechEnd {
                        start_timestamp_ms: start_1,
                        end_timestamp_ms: end_1,
                        ..
                    },
                    VadTransition::SpeechEnd {
                        start_timestamp_ms: start_2,
                        end_timestamp_ms: end_2,
                        ..
                    },
                ) => {
                    if start_1 != start_2 {
                        return false;
                    } else if end_1 != end_2
                        && (*end_1 as isize - *end_2 as isize).abs() as usize
                            != allowed_post_pad_diff
                    {
                        return false;
                    }
                }
                _ => unreachable!(),
            }
        }

        true
    }

    fn eq_current_session_samples(
        &self,
        other: &Self,
        _self_config: &VadConfig,
        _other_config: &VadConfig,
    ) -> bool {
        self.current_session_samples == other.current_session_samples
    }

    fn eq_current_speech_samples(
        &self,
        other: &Self,
        self_config: &VadConfig,
        other_config: &VadConfig,
    ) -> bool {
        // Some differences are introduced by modified config and they are expected.
        let allowed_post_pad_diff = (self_config.post_speech_pad.as_millis() as isize
            - other_config.post_speech_pad.as_millis() as isize)
            .abs() as usize
            * self_config.sample_rate
            / 1000;

        for (baseline, current) in self
            .current_speech_samples
            .iter()
            .zip(other.current_speech_samples.iter())
        {
            if baseline != current
                && (*baseline as isize - *current as isize).abs() as usize != allowed_post_pad_diff
            {
                return false;
            }
        }

        true
    }

    fn eq_current_silence_samples(
        &self,
        other: &Self,
        _self_config: &VadConfig,
        _other_config: &VadConfig,
    ) -> bool {
        self.current_silence_samples == other.current_silence_samples
    }

    fn eq_likelihoods(
        &self,
        other: &Self,
        _self_config: &VadConfig,
        _other_config: &VadConfig,
    ) -> bool {
        self.likelihoods == other.likelihoods
    }
}

#[test]
// #[traced_test]
fn chunk_50_default_params_16k() {
    let mut config = VadConfig::default();
    run_snapshot_test(50, &config, "default");

    config.post_speech_pad = Duration::from_millis(100);
    run_snapshot_test(50, &config, "default");
}

#[test]
// #[traced_test]
fn chunk_50_default_params_8k() {
    let mut config = VadConfig::default();
    config.sample_rate = 8000;
    run_snapshot_test(50, &config, "default");

    config.post_speech_pad = Duration::from_millis(100);
    run_snapshot_test(50, &config, "default");
}

#[test]
// #[traced_test]
fn chunk_30_default_params_16k() {
    let mut config = VadConfig::default();
    run_snapshot_test(30, &config.clone(), "default");

    config.post_speech_pad = Duration::from_millis(100);
    run_snapshot_test(30, &config, "default");
}

#[test]
// #[traced_test]
fn chunk_30_default_params_8k() {
    let mut config = VadConfig::default();
    config.sample_rate = 8000;
    run_snapshot_test(30, &config, "default");

    config.post_speech_pad = Duration::from_millis(100);
    run_snapshot_test(30, &config, "default");
}

#[test]
// #[traced_test]
fn chunk_20_default_params_16k() {
    let mut config = VadConfig::default();
    run_snapshot_test(20, &config, "default");

    config.post_speech_pad = Duration::from_millis(100);
    run_snapshot_test(20, &config, "default");
}

#[test]
// #[traced_test]
fn chunk_20_default_params_8k() {
    let mut config = VadConfig::default();
    config.sample_rate = 8000;
    run_snapshot_test(20, &config, "default");

    config.post_speech_pad = Duration::from_millis(100);
    run_snapshot_test(20, &config, "default");
}

fn run_snapshot_test(chunk_ms: usize, config: &VadConfig, config_name: &str) {
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
        config: config.clone(),
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
    // println!("Checking snapshot is generated with same configuration!");
    // compare_configs(&summary.config, &expected.config);
    assert_eq!(summary.input_size_ms, expected.input_size_ms);

    let failures = expected.eq(&summary);

    if let Some(failures) = failures {
        println!();
        println!("You have some failing files and targets. If you get a snapshot file and audio you can plot it via our plot_audio script e.g.");
        println!();
        println!(
            "python3 scripts/plot_audio.py -a {} -i {}",
            baseline_report.display(),
            current_report.display()
        );
        println!();

        let mut failure_string = String::from("\n");
        for audio_file in failures.keys() {
            failure_string.push_str(&format!("{:?}:\n", audio_file));
            for actual_failure in failures[audio_file].iter() {
                failure_string.push_str(&format!("\t{}\n", actual_failure));
            }
        }
        panic!("The following failures were detected: {}", failure_string);
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
    assert!(!result.is_empty());
    result
}

fn get_chunk_size(sample_rate: usize, chunk_ms: usize) -> usize {
    (sample_rate * chunk_ms) / 1000
}

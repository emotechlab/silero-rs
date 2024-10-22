use rubato::{FftFixedIn, Resampler};
use tracing::trace;

/// Resample one channel of pcm data into desired sample rate.
pub fn resample_pcm(
    pcm_data: Vec<f32>,
    original_sample_rate: usize,
    desired_sample_rate: usize,
) -> anyhow::Result<Vec<f32>> {
    if original_sample_rate == desired_sample_rate {
        trace!("no need to do any resample work");
        return Ok(pcm_data);
    }
    trace!(
        "Resampling {} samples from {}Hz to {}Hz",
        &pcm_data.len(),
        original_sample_rate,
        desired_sample_rate
    );

    let mut resampler = FftFixedIn::new(
        original_sample_rate,
        desired_sample_rate,
        pcm_data.len(),
        pcm_data.len(), // I don't know what does sub_chunks mean, just a random choice.
        1,
    )?;

    let waves_in = vec![pcm_data];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(waves_out[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use hound::WavReader;
    use std::fs::read_dir;
    use std::path::{Path, PathBuf};

    fn read_wav_file(file_path: impl AsRef<Path>, sample_rate: usize) -> Result<Vec<f32>> {
        let step = if sample_rate == 16000 { 1 } else { 2 };

        let samples: Vec<f32> = WavReader::open(file_path)?
            .into_samples()
            .step_by(step)
            .map(|x| {
                let x_f32 = x.unwrap_or(0i16) as f32 / i16::MAX as f32;
                x_f32.clamp(-1.0, 1.0)
            })
            .collect();

        Ok(samples)
    }

    fn get_audios() -> Vec<PathBuf> {
        let audio_dir = Path::new("tests/audio/");
        let mut result = vec![];
        for entry in read_dir(&audio_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if !path.is_dir() && path.extension().unwrap() == "wav" {
                result.push(path.to_path_buf());
            }
        }
        result
    }

    #[test]
    fn read_wav() {
        let audios = get_audios();
        assert!(!audios.is_empty());
        for audio in audios {
            let samples = read_wav_file(audio, 16000).unwrap();
            assert!(!samples.is_empty());
        }
    }

    #[test]
    fn resample_up() {
        let audios = get_audios();
        assert!(!audios.is_empty());

        for audio in audios {
            let samples_16k = read_wav_file(audio, 16000).unwrap();
            let samples_32k = resample_pcm(samples_16k.clone(), 16000, 32000).unwrap();
            assert_eq!(samples_16k.len() * 2, samples_32k.len());
        }
    }

    #[test]
    fn resample_no_change() {
        let audios = get_audios();
        assert!(!audios.is_empty());

        for audio in audios {
            let samples_16k = read_wav_file(audio, 16000).unwrap();
            let resampled_16k = resample_pcm(samples_16k.clone(), 16000, 16000).unwrap();
            assert_eq!(samples_16k.len(), resampled_16k.len());
        }
    }

    #[test]
    fn resample_down() {
        let audios = get_audios();
        assert!(!audios.is_empty());

        for audio in audios {
            let samples_16k = read_wav_file(audio, 16000).unwrap();
            let samples_8k = resample_pcm(samples_16k.clone(), 16000, 8000).unwrap();
            assert_eq!(samples_16k.len() / 2, samples_8k.len());
        }
    }
}

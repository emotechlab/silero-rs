use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tracing::info;

/// Resample one channel of pcm data into desired sample rate.
pub fn resample_pcm(
    pcm_data: Vec<f32>,
    original_sample_rate: usize,
    desired_sample_rate: usize,
) -> anyhow::Result<Vec<f32>> {
    info!(
        "Resampling {} samples from {}Hz to {}Hz",
        &pcm_data.len(),
        original_sample_rate,
        desired_sample_rate
    );
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        desired_sample_rate as f64 / original_sample_rate as f64,
        2.0,
        params,
        pcm_data.len(),
        1,
    )?;

    let waves_in = vec![pcm_data];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(waves_out[0].clone())
}

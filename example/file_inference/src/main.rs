//! An example to show how to read any audio file into pcm format and then send it to silero vad for
//! inference.

use anyhow::Result;
use audio_reading::{MultiChannelStrategy, read_audio};
use clap::Parser;
use silero::{VadConfig, VadSession};

mod audio_reading;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(short, long)]
    file: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let (samples, original_sample_rate) = read_audio(args.file, MultiChannelStrategy::FirstOnly);
    
    let mut config = VadConfig::default();
    config.sample_rate = original_sample_rate;
    let mut session = VadSession::new(config)?;
    
    // You do not need to worry about sample rate as long as you are using our library with
    // `audio_resampler` feature enabled.
    let results = session.process(&samples)?;
    dbg!(&results);
    
    Ok(())
}

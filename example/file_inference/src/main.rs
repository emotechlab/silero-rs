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
    let samples = read_audio(args.file, 16000, MultiChannelStrategy::FirstOnly);
    
    let config = VadConfig::default();
    let mut session = VadSession::new(config)?;
    
    let results = session.process(&samples)?;
    dbg!(&results);
    
    Ok(())
}

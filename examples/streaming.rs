use silero::{Session, SpeechOptions, SpeechSegmenter, StreamState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut session = Session::bundled()?;
  let config = SpeechOptions::default();
  let mut stream = StreamState::new(config.sample_rate());
  let mut segmenter = SpeechSegmenter::new(config);

  let synthetic_audio = vec![0.0_f32; config.sample_rate().chunk_samples() * 8];
  segmenter.process_samples(&mut session, &mut stream, &synthetic_audio, |segment| {
    println!(
      "segment {:.2}s -> {:.2}s",
      segment.start_seconds(),
      segment.end_seconds()
    );
  })?;
  segmenter.finish_stream(&mut session, &mut stream, |segment| {
    println!(
      "segment {:.2}s -> {:.2}s",
      segment.start_seconds(),
      segment.end_seconds()
    );
  })?;

  Ok(())
}

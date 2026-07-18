use silero::{Session, SpeechOptions, SpeechSegmenter, SpeechSegmenterExt, StreamState};

const MODEL_BYTES: &[u8] = include_bytes!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/models/silero_vad.onnx"
));

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut session = Session::from_memory(MODEL_BYTES)?;
  let config = SpeechOptions::default();
  let mut stream = StreamState::new(config.sample_rate());
  let mut segmenter = SpeechSegmenter::new(config.clone());

  let synthetic_audio = vec![0.0_f32; config.sample_rate().chunk_samples() * 8];
  let print = |segment: silero::SpeechSegment| {
    println!(
      "segment {:.2}s -> {:.2}s",
      segment.start_seconds(),
      segment.end_seconds()
    );
  };

  if let Some(segment) = segmenter.push_samples(&mut session, &mut stream, &synthetic_audio)? {
    print(segment);
    while let Some(more) = segmenter.push_samples(&mut session, &mut stream, &[])? {
      print(more);
    }
  }
  if let Some(segment) = segmenter.finish_stream(&mut session, &mut stream)? {
    print(segment);
    while let Some(more) = segmenter.push_samples(&mut session, &mut stream, &[])? {
      print(more);
    }
  }

  Ok(())
}

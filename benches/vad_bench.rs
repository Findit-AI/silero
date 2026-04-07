//! Criterion benchmarks for the Silero VAD Rust port.
//!
//! Measures three things:
//!
//! 1. **Raw model throughput** — single-chunk `SileroModel::process`
//!    latency, reflecting the ONNX inference cost per 32 ms window.
//! 2. **Batch `speech_timestamps`** throughput on a realistic 10 s
//!    speech buffer.
//! 3. **Streaming `VadIterator`** throughput, same 10 s buffer,
//!    measuring per-chunk state-machine overhead.
//!
//! Run from the crate root:
//!
//! ```bash
//! RUSTFLAGS="-L /path/to/onnxruntime" \
//!     cargo bench --features wav
//! ```
//!
//! Criterion writes HTML reports to `target/criterion/`. If the
//! reference fixture audio is not present, the bench falls back to a
//! synthetic 10 s noise buffer.

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use silero::{speech_timestamps_with_model, SampleRate, SileroModel, VadConfig, VadIterator};

fn model_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx")
}

/// Build a 10-second synthetic mono 16 kHz buffer. Content doesn't
/// matter for latency measurements — Silero's inference cost is
/// independent of the actual probability output.
fn synthetic_audio_10s() -> Vec<f32> {
  // A mix of sine + tiny noise so probabilities aren't stuck at 0.
  let n = 10 * 16_000;
  (0..n)
    .map(|i| {
      let t = i as f32 / 16_000.0;
      let tone = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.1;
      let noise = ((i as u32).wrapping_mul(2654435769) >> 16) as f32 / 65535.0 - 0.5;
      tone + noise * 0.01
    })
    .collect()
}

fn bench_model_process(c: &mut Criterion) {
  let path = model_path();
  if !path.exists() {
    eprintln!("Skipping model_process bench: {} not found", path.display());
    return;
  }
  let mut model = SileroModel::from_path(&path).expect("model load");
  let chunk = vec![0.05_f32; SampleRate::Rate16k.chunk_samples()];

  let mut g = c.benchmark_group("model_process");
  g.throughput(Throughput::Elements(1));
  g.bench_function("16k_512samples", |b| {
    b.iter(|| {
      std::hint::black_box(model.process(std::hint::black_box(&chunk)).unwrap());
    });
  });
  g.finish();
}

fn bench_speech_timestamps(c: &mut Criterion) {
  let path = model_path();
  if !path.exists() {
    eprintln!(
      "Skipping speech_timestamps bench: {} not found",
      path.display()
    );
    return;
  }
  let mut model = SileroModel::from_path(&path).expect("model load");
  let audio = synthetic_audio_10s();

  let mut g = c.benchmark_group("speech_timestamps");
  g.throughput(Throughput::Elements(audio.len() as u64));
  g.sample_size(20); // each call runs ~313 inferences; default 100 is overkill
  g.bench_function("10s_batch", |b| {
    b.iter(|| {
      std::hint::black_box(
        speech_timestamps_with_model(
          &mut model,
          std::hint::black_box(&audio),
          16_000,
          VadConfig::default(),
        )
        .unwrap(),
      );
    });
  });
  g.finish();
}

fn bench_streaming_iter(c: &mut Criterion) {
  let path = model_path();
  if !path.exists() {
    eprintln!(
      "Skipping streaming_iter bench: {} not found",
      path.display()
    );
    return;
  }
  let audio = synthetic_audio_10s();
  let chunk_size = SampleRate::Rate16k.chunk_samples();

  let mut g = c.benchmark_group("streaming_iter");
  g.throughput(Throughput::Elements(audio.len() as u64));
  g.sample_size(20);
  g.bench_function("10s_stream", |b| {
    let mut vad = VadIterator::new(&path, VadConfig::default()).expect("load");
    b.iter(|| {
      vad.reset();
      for chunk in audio.chunks_exact(chunk_size) {
        std::hint::black_box(vad.process(std::hint::black_box(chunk)).unwrap());
      }
    });
  });
  g.finish();
}

criterion_group!(
  benches,
  bench_model_process,
  bench_speech_timestamps,
  bench_streaming_iter
);
criterion_main!(benches);

//! `silero-parity-runner` — load a 16 kHz mono WAV via `ffmpeg-next`,
//! push it through `silero::detect_speech` (the production one-shot
//! offline path), and dump the resulting speech segments as JSON.
//! Pair with `python/silero_vad_runner.py` (same JSON schema,
//! `runner = "silero-vad-py"`) and `python/score.py` for IoU
//! comparison.
//!
//! This binary is **NOT** part of `cargo test`. It's invoked from the
//! `run.sh` driver. Audio loading uses `ffmpeg-next` so the f32 buffer
//! the silero ONNX model consumes is byte-identical to what the upstream
//! Python `silero-vad` package consumes (which also goes through
//! ffmpeg via `torchaudio` / `read_audio`).
//!
//! All `SpeechOptions` knobs are exposed via flags so the run.sh
//! driver can pass parameters that match the Python runner exactly.
//! Defaults match the silero crate's `SpeechOptions::default()`, which
//! in turn match upstream silero-vad PyPI defaults (threshold 0.5,
//! min_speech_duration_ms 250, min_silence_duration_ms 100,
//! speech_pad_ms 30, min_silence_at_max_speech_ms 98).

use std::{
  fs,
  io::Write,
  path::{Path, PathBuf},
  sync::Once,
  time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use serde_json::json;
use sha2::{Digest, Sha256};
use silero::{SampleRate, Session, SpeechOptions, detect_speech};

const SILERO_CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
// SHA-256 of the bundled ONNX model bytes. Computed on demand below.
// Logged so a snapshot rebuild in the silero crate that swaps
// `models/silero_vad.onnx` cannot silently invalidate the parity
// numbers — the JSON output records exactly which model bytes ran.

#[derive(Parser, Debug)]
#[command(
  about = "Run silero (Rust crate) VAD on a 16 kHz mono WAV; emit JSON for side-by-side comparison with upstream Python silero-vad."
)]
struct Args {
  /// Path to a 16 kHz mono WAV (or any audio container ffmpeg can
  /// decode; resampled to 16 kHz mono internally).
  wav_path: PathBuf,

  /// Output file (defaults to stdout).
  #[arg(long)]
  out: Option<PathBuf>,

  /// Speech-onset probability threshold. Silero crate default: 0.5.
  #[arg(long, default_value_t = 0.5)]
  threshold: f32,

  /// Minimum speech duration in milliseconds; shorter speech bursts are
  /// dropped. Silero crate default: 250.
  #[arg(long, default_value_t = 250)]
  min_speech_ms: u64,

  /// Minimum silence duration in milliseconds before a speech segment
  /// is closed. Silero crate default: 100.
  #[arg(long, default_value_t = 100)]
  min_silence_ms: u64,

  /// Speech padding (added at both ends of every emitted segment) in
  /// milliseconds. Silero crate default: 30.
  #[arg(long, default_value_t = 30)]
  speech_pad_ms: u64,

  /// Minimum silence used as a preferred split point when
  /// `--max-speech-s` is hit, in milliseconds. Silero crate default: 98
  /// (which matches upstream Python silero-vad's 0.098 s default).
  #[arg(long, default_value_t = 98)]
  min_silence_at_max_speech_ms: u64,

  /// Maximum speech duration in seconds before the segmenter
  /// force-splits a long segment. Defaults to "no limit" (matches both
  /// the Rust crate and Python silero-vad defaults). Pass e.g. `30` to
  /// match WhisperX-style chunking.
  #[arg(long)]
  max_speech_s: Option<f64>,
}

/// Idempotent guard for `ffmpeg::init()`. Mirrors the whispery parity
/// runner's pattern.
fn ffmpeg_init() -> Result<()> {
  static INIT: Once = Once::new();
  let mut init_err: Option<ffmpeg::Error> = None;
  INIT.call_once(|| {
    if let Err(e) = ffmpeg::init() {
      init_err = Some(e);
    }
  });
  if let Some(e) = init_err {
    Err(anyhow::anyhow!("ffmpeg::init failed: {e}"))
  } else {
    Ok(())
  }
}

/// Load an audio file as 16 kHz mono f32 via ffmpeg-next.
///
/// This mirrors the loader in `whispery`'s parity runner. Decoding
/// path: container open → audio decoder → resample to 16 kHz mono
/// `s16` (signed 16-bit packed, little-endian) → cast each sample to
/// `f32` and divide by exactly `32768.0`.
///
/// Why s16-then-divide rather than f32-direct: upstream Python
/// silero-vad loads audio via `torchaudio.load` (or `whisperx.audio`'s
/// ffmpeg shell-out) which lands on `np.float32 / 32768.0`. Doing the
/// same conversion on the Rust side keeps the f32 buffer the model
/// sees byte-identical, so a hash comparison on the JSON output's
/// `clip_sha256` field can verify both runners decoded the audio the
/// same way before flagging any output divergence as a model issue.
///
/// Returns `(samples, duration_s, sha256)`.
fn read_audio_16k_mono_f32(path: &Path) -> Result<(Vec<f32>, f64, String)> {
  use ffmpeg::format::sample::{Sample, Type as SampleType};
  use ffmpeg::software::resampling::Context as Resampler;
  use ffmpeg::{ChannelLayout, codec::context::Context as CodecContext, frame, media};

  ffmpeg_init()?;

  let mut ictx = ffmpeg::format::input(path)
    .with_context(|| format!("open audio container at {}", path.display()))?;
  let stream = ictx
    .streams()
    .best(media::Type::Audio)
    .ok_or_else(|| anyhow::anyhow!("{}: no audio stream", path.display()))?;
  let stream_index = stream.index();

  let codec_ctx = CodecContext::from_parameters(stream.parameters())
    .with_context(|| format!("decoder context for {}", path.display()))?;
  let mut decoder = codec_ctx
    .decoder()
    .audio()
    .with_context(|| format!("audio decoder for {}", path.display()))?;
  decoder
    .set_parameters(stream.parameters())
    .with_context(|| format!("decoder set_parameters for {}", path.display()))?;

  const TARGET_RATE: u32 = 16_000;
  let target_format = Sample::I16(SampleType::Packed);
  let target_layout = ChannelLayout::MONO;

  // PCM/WAV decoders commonly emit frames with `ch_layout.order =
  // UNSPEC` (only the channel count is set); libswresample's
  // `swr_alloc_set_opts2` rejects that in FFmpeg 7+. Fall back to
  // `ChannelLayout::default(channels)` if the source layout is empty.
  let resolve_src_layout =
    |layout: ChannelLayout, channels: i32| -> ChannelLayout {
      if layout.is_empty() {
        ChannelLayout::default(channels)
      } else {
        layout
      }
    };

  let mut src_format = decoder.format();
  let mut src_rate = decoder.rate();
  let mut src_layout = resolve_src_layout(decoder.channel_layout(), decoder.channels() as i32);

  let build_resampler = |src_format,
                         src_layout,
                         src_rate|
   -> Result<Resampler> {
    Resampler::get(
      src_format,
      src_layout,
      src_rate,
      target_format,
      target_layout,
      TARGET_RATE,
    )
    .with_context(|| format!("init libswresample for {}", path.display()))
  };

  let mut resampler = build_resampler(src_format, src_layout, src_rate)?;

  let mut samples_f32: Vec<f32> = Vec::new();
  let mut decoded = frame::Audio::empty();

  // Push i16 samples from a packed-mono frame into `samples_f32`,
  // dividing by the literal `32768.0` exactly as
  // WhisperX/torchaudio does.
  let push_resampled = |frame: &frame::Audio, dst: &mut Vec<f32>| {
    let n = frame.samples();
    if n == 0 {
      return;
    }
    let plane: &[i16] = frame.plane::<i16>(0);
    debug_assert!(plane.len() >= n);
    dst.reserve(n);
    for &s in &plane[..n] {
      dst.push(s as f32 / 32768.0_f32);
    }
  };

  // Run a decoded frame through the resampler. Handles
  // `InputChanged` / `OutputChanged` by rebuilding the resampler
  // against the new source params.
  let run_resample = |decoded: &frame::Audio,
                      resampler: &mut Resampler,
                      samples_f32: &mut Vec<f32>,
                      src_format: &mut Sample,
                      src_layout: &mut ChannelLayout,
                      src_rate: &mut u32|
   -> Result<()> {
    let mut resampled = frame::Audio::empty();
    match resampler.run(decoded, &mut resampled) {
      Ok(_) => {
        push_resampled(&resampled, samples_f32);
      }
      Err(ffmpeg::Error::InputChanged | ffmpeg::Error::OutputChanged) => {
        *src_format = decoded.format();
        *src_layout = resolve_src_layout(
          decoded.channel_layout(),
          decoded.channels() as i32,
        );
        *src_rate = decoded.rate();
        *resampler = build_resampler(*src_format, *src_layout, *src_rate)?;
        let mut retried = frame::Audio::empty();
        resampler
          .run(decoded, &mut retried)
          .context("libswresample::run after rebuild")?;
        push_resampled(&retried, samples_f32);
      }
      Err(e) => return Err(anyhow::anyhow!("libswresample::run: {e}")),
    }
    Ok(())
  };

  let fixup_frame_layout = |frame: &mut frame::Audio, src_layout: ChannelLayout| {
    if frame.channel_layout().is_empty() {
      frame.set_channel_layout(src_layout);
    }
  };

  for (s, packet) in ictx.packets() {
    if s.index() != stream_index {
      continue;
    }
    decoder.send_packet(&packet).context("decoder.send_packet")?;
    while decoder.receive_frame(&mut decoded).is_ok() {
      fixup_frame_layout(&mut decoded, src_layout);
      run_resample(
        &decoded,
        &mut resampler,
        &mut samples_f32,
        &mut src_format,
        &mut src_layout,
        &mut src_rate,
      )?;
    }
  }
  decoder.send_eof().context("decoder.send_eof")?;
  while decoder.receive_frame(&mut decoded).is_ok() {
    fixup_frame_layout(&mut decoded, src_layout);
    run_resample(
      &decoded,
      &mut resampler,
      &mut samples_f32,
      &mut src_format,
      &mut src_layout,
      &mut src_rate,
    )?;
  }

  // Final libswresample flush. `OutputChanged` here means "no buffered
  // samples" in the rate-1:1 case (which is what the dia 16 kHz mono
  // PCM fixtures hit). Treat it as a no-op rather than a hard error.
  loop {
    let mut tail = frame::Audio::empty();
    match resampler.flush(&mut tail) {
      Ok(_) => {
        if tail.samples() == 0 {
          break;
        }
        push_resampled(&tail, &mut samples_f32);
      }
      Err(ffmpeg::Error::OutputChanged) => break,
      Err(e) => {
        return Err(anyhow::anyhow!("libswresample::flush at EOF: {e}"));
      }
    }
  }

  if samples_f32.is_empty() {
    bail!(
      "{}: ffmpeg-next decoded zero samples; corrupt or empty audio?",
      path.display()
    );
  }

  let duration_s = samples_f32.len() as f64 / TARGET_RATE as f64;

  // Hash the f32 bytes (LE) the model will see — same trick the
  // whispery harness uses. Comparing this against the Python runner's
  // own clip_sha256 is what catches loader-quantisation divergences.
  let mut hasher = Sha256::new();
  // Safety: `f32` is `Copy + 'static`, layout is well-defined as 4
  // little-endian bytes per sample on every target this harness ships
  // to (macOS / Linux x86_64+aarch64).
  let bytes = unsafe {
    std::slice::from_raw_parts(
      samples_f32.as_ptr() as *const u8,
      samples_f32.len() * std::mem::size_of::<f32>(),
    )
  };
  hasher.update(bytes);
  let sha = format!("{:x}", hasher.finalize());

  Ok((samples_f32, duration_s, sha))
}

fn model_sha256() -> String {
  let mut hasher = Sha256::new();
  hasher.update(silero::BUNDLED_MODEL);
  format!("{:x}", hasher.finalize())
}

fn main() -> Result<()> {
  let args = Args::parse();

  let (samples, duration_s, clip_sha256) = read_audio_16k_mono_f32(&args.wav_path)?;
  eprintln!(
    "[silero-parity] wav={} dur={:.2}s samples={} sha256={}",
    args.wav_path.display(),
    duration_s,
    samples.len(),
    &clip_sha256[..16]
  );

  // Build SpeechOptions from CLI flags. Every default matches the
  // silero crate's `SpeechOptions::default()`, which in turn matches
  // upstream Python silero-vad defaults.
  let mut opts = SpeechOptions::new()
    .with_sample_rate(SampleRate::Rate16k)
    .with_start_threshold(args.threshold)
    .with_min_speech_duration(Duration::from_millis(args.min_speech_ms))
    .with_min_silence_duration(Duration::from_millis(args.min_silence_ms))
    .with_speech_pad(Duration::from_millis(args.speech_pad_ms))
    .with_min_silence_at_max_speech(Duration::from_millis(
      args.min_silence_at_max_speech_ms,
    ));
  if let Some(s) = args.max_speech_s {
    let ms = (s * 1000.0).round() as u64;
    opts = opts.with_max_speech_duration(Duration::from_millis(ms));
  }

  eprintln!(
    "[silero-parity] threshold={} min_speech_ms={} min_silence_ms={} \
     pad_ms={} min_silence_at_max_speech_ms={} max_speech_s={:?}",
    args.threshold,
    args.min_speech_ms,
    args.min_silence_ms,
    args.speech_pad_ms,
    args.min_silence_at_max_speech_ms,
    args.max_speech_s,
  );

  let mut session = Session::bundled().context("load bundled silero ONNX session")?;
  let segments = detect_speech(&mut session, &samples, opts).context("detect_speech")?;

  eprintln!(
    "[silero-parity] {} segments detected",
    segments.len()
  );

  let segments_json: Vec<serde_json::Value> = segments
    .iter()
    .map(|s| {
      json!({
        "start_s": s.start_seconds(),
        "end_s": s.end_seconds(),
        "start_sample": s.start_sample(),
        "end_sample": s.end_sample(),
      })
    })
    .collect();

  let payload = json!({
    "runner": "silero-rs",
    "silero_crate_version": SILERO_CRATE_VERSION,
    "model_sha256": model_sha256(),
    "clip_path": args.wav_path.display().to_string(),
    "clip_sha256": clip_sha256,
    "duration_s": duration_s,
    "params": {
      "threshold": args.threshold,
      "min_speech_duration_ms": args.min_speech_ms,
      "min_silence_duration_ms": args.min_silence_ms,
      "speech_pad_ms": args.speech_pad_ms,
      "min_silence_at_max_speech_ms": args.min_silence_at_max_speech_ms,
      "max_speech_s": args.max_speech_s,
      "sampling_rate": 16_000,
      "window_size_samples": 512,
    },
    "segment_count": segments.len(),
    "segments": segments_json,
  });

  let serialized = serde_json::to_string_pretty(&payload)?;
  match args.out {
    Some(path) => {
      let mut f = fs::File::create(&path)
        .with_context(|| format!("create output {}", path.display()))?;
      f.write_all(serialized.as_bytes())?;
      f.write_all(b"\n")?;
      eprintln!(
        "[silero-parity] wrote {} segments to {}",
        segments.len(),
        path.display()
      );
    }
    None => {
      println!("{serialized}");
    }
  }

  Ok(())
}

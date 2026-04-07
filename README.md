# silero

A Rust implementation of [Silero VAD] (Voice Activity Detection),
faithfully ported from the official Python reference at
[`snakers4/silero-vad`][upstream].

[Silero VAD]: https://github.com/snakers4/silero-vad
[upstream]: https://github.com/snakers4/silero-vad

The neural network is unchanged — we load the same `silero_vad.onnx`
model (~2.3 MB) via [`ort`] and run inference in Rust. Everything
*around* the model (preprocessing, rolling LSTM state, streaming
boundary events, padding, segment timestamp computation) is a
line-by-line port of the upstream Python so that the same audio
produces identical results in both implementations.

[`ort`]: https://docs.rs/ort

---

## Why a Rust port

The upstream project ships a working Python reference, but a few
things make it hard to use from a Rust application:

1. **Embedding cost.** Python + PyTorch + torchaudio is a 5–10 GB
   dependency. ONNX-via-ORT in Rust is ~30 MB total.
2. **Throughput.** A pure-Rust pipeline (audio decode → resample →
   VAD → STT) avoids the GIL and the Python ↔ Rust FFI bridge.
3. **Hidden ONNX subtlety.** The ONNX export expects a `[1, 64+512]`
   audio tensor — 64 samples of rolling context prepended to each
   chunk. Passing the raw 512-sample chunk silently triggers an
   internal "too short" branch and produces near-zero probabilities
   for every input. This is documented only in the Python
   `OnnxWrapper` source. We've solved it once so you don't have to.

The model itself is identical to upstream; the accuracy and recall
numbers below match the Python reference within floating-point
tolerance.

---

## Quick start

```toml
[dependencies]
silero = { git = "https://github.com/Findit-AI/silero" }
```

```rust
use silero::{speech_timestamps, SampleRate, VadConfig};

// Load mono f32 PCM at 16 kHz from somewhere.
let audio: Vec<f32> = read_my_audio();

let segments = silero::speech_timestamps(
    "models/silero_vad.onnx",
    &audio,
    16_000,
    VadConfig::default(),
)?;

for segment in segments {
    println!(
        "speech {:.2}s → {:.2}s",
        segment.start_seconds(SampleRate::Rate16k),
        segment.end_seconds(SampleRate::Rate16k),
    );
}
# Ok::<_, silero::SileroError>(())
```

## Three layers of API

Pick the lowest-level layer that still does what you need:

| Layer | Type | Use case |
|---|---|---|
| `speech_timestamps()` | one-shot batch function | full audio in memory, want a list of segments out (offline indexing, transcription pre-filter) |
| `VadIterator` | streaming, event-based | live audio, want immediate Start/End events with sub-second latency |
| `SileroModel` | raw ONNX wrapper | per-chunk probability inspection, custom boundary logic, batching across streams |

### Batch API — `speech_timestamps`

```rust
use silero::{speech_timestamps, VadConfig};

let segments = speech_timestamps(
    "models/silero_vad.onnx",
    &audio_16k,
    16_000,
    VadConfig::default()
        .with_min_silence_ms(200)
        .with_speech_pad_ms(50),
)?;

println!("found {} speech regions", segments.len());
# Ok::<_, silero::SileroError>(())
```

This is the equivalent of upstream Python's `get_speech_timestamps`,
including the same default tuning (`threshold=0.5`,
`min_speech_duration_ms=250`, `min_silence_duration_ms=100`,
`speech_pad_ms=30`) and the same complex padding rules at segment
boundaries.

### Streaming API — `VadIterator`

```rust
use silero::{Event, VadConfig, VadIterator};

let mut vad = VadIterator::new("models/silero_vad.onnx", VadConfig::default())?;

// Feed every 512-sample chunk from the live source.
for chunk in incoming_audio.chunks_exact(vad.chunk_samples()) {
    if let Some(event) = vad.process(chunk)? {
        match event {
            Event::Start(sample) => println!("speech started @ {sample}"),
            Event::End(sample) => println!("speech ended   @ {sample}"),
        }
    }
}

// Flush any speech region still open at end of stream.
if let Some(Event::End(sample)) = vad.finish() {
    println!("flushed end @ {sample}");
}
# Ok::<_, silero::SileroError>(())
```

The streaming state machine is a one-to-one port of upstream Python's
`VADIterator`, with one deliberate addition: `finish()` returns a
synthetic `End` event for any region still open at end of stream
(the upstream silently drops these). Everything else matches the
reference exactly.

### Low-level API — `SileroModel`

```rust
use silero::{SampleRate, SileroModel};

let mut model = SileroModel::from_path("models/silero_vad.onnx")?;

let chunk = vec![0.0_f32; SampleRate::Rate16k.chunk_samples()]; // 512
let prob = model.process(&chunk)?;
println!("voice prob: {prob:.3}");

// Reset between independent streams.
model.reset();
# Ok::<_, silero::SileroError>(())
```

`SileroModel` is the thinnest possible wrapper: one method for
inference, one for state reset, and that's it. Use it when you want
to drive your own boundary logic or feed the same loaded model to
multiple iterators.

## Defaults match upstream

Every default value in `VadConfig` matches the upstream Python
`get_speech_timestamps` defaults:

| Parameter | Default | Source |
|---|---|---|
| `threshold` | `0.5` | upstream |
| `negative_threshold` | `max(threshold - 0.15, 0.01)` (lazy) | upstream |
| `min_speech_duration_ms` | `250` | upstream |
| `min_silence_duration_ms` | `100` | upstream |
| `speech_pad_ms` | `30` | upstream |
| `min_silence_at_max_speech_ms` | `98` | upstream |
| `max_speech_duration_ms` | effectively infinity | upstream |
| sample rate | `16 kHz` | upstream |

## Sample rates

Silero VAD is trained at exactly 8 kHz and 16 kHz. The
`speech_timestamps` function additionally accepts integer multiples
of 16 kHz (32, 48, 96 kHz, …) and downsamples them by naive
decimation, matching upstream Python. Anything else (44.1 kHz, 22.05
kHz, …) must be resampled by the caller. The optional `resample`
feature flag enables a high-quality resampler via `rubato`:

```toml
[dependencies]
silero = { git = "...", features = ["resample"] }
```

```rust
let audio_16k = silero::resample::resample(&audio_44k, 44_100, 16_000)?;
```

## Audio I/O

The core crate has zero audio-I/O dependencies. The optional `wav`
feature enables a small `read_wav` helper for quick scripts:

```toml
silero = { git = "...", features = ["wav"] }
```

```rust
let (audio, sample_rate) = silero::io::read_wav("recording.wav")?;
```

For production audio decoding (MP3, AAC, OGG, multi-channel mixing,
…), use a proper crate like [`symphonia`] or decode upstream in your
application.

[`symphonia`]: https://github.com/pdeljanov/Symphonia

## Examples

```bash
# Batch detection on a WAV file
cargo run --release --example detect_file --features wav -- recording.wav

# Streaming events from a WAV file (must be 16 kHz mono)
cargo run --release --example streaming --features wav -- recording.wav
```

## Benchmarks

Run on Apple Silicon (M-series), 16 kHz mono input, single thread:

| Workload | Latency | Throughput |
|---|---|---|
| `SileroModel::process` (one 32 ms chunk) | ~120 µs | ~270× real-time per call |
| `speech_timestamps` (10 s buffer) | ~37 ms | ~270× real-time |
| `VadIterator` (10 s stream) | ~37 ms | ~270× real-time |

Reproduce with:

```bash
cargo bench
```

## Accuracy

Verified against the reference implementation on five datasets
(silence, white noise, environmental sounds, balanced two-person
dialog, monologue, music mix), totalling ~127 minutes of audio.
Speech ratios match within floating-point tolerance, including the
known intentional Silero behaviour of treating sung vocals as
non-speech (which is the right thing for STT pre-filtering).

## What this crate does *not* do

- **Re-implement the neural network.** The Conv / LSTM / sigmoid
  operations stay in ONNX Runtime, which is hand-optimised. A
  from-scratch implementation would not be faster and would multiply
  the maintenance burden.
- **Detect sung vocals.** Silero is trained for spoken speech, not
  vocal music. Pop, rap, and operatic vocals are intentionally
  classified as non-speech. For music use cases, consider a music
  classifier or a CLAP-style audio-text model.
- **Audio decoding.** Pass us decoded f32 PCM. The optional `wav`
  feature is intentionally minimal — anything beyond it is out of
  scope.

## Comparison vs the upstream Python reference

| Property | upstream Python | this crate |
|---|---|---|
| Same neural network | ✓ | ✓ (same `silero_vad.onnx`) |
| Same defaults | ✓ | ✓ |
| Streaming API | `VADIterator` | `VadIterator` |
| Batch API | `get_speech_timestamps` | `speech_timestamps` |
| Padding rules | as documented | identical port |
| Max-speech splitting | as documented | identical port |
| Auto-decimation of 32/48 kHz | ✓ | ✓ |
| Auto-emit end on stream close | ✗ | ✓ via `finish()` |
| Embedded ONNX file | ✓ (~2.3 MB) | ✓ (`models/silero_vad.onnx`) |
| Runtime dependencies | Python + torch + torchaudio | `ort` + `ndarray` |
| Single-binary deployment | ✗ | ✓ |

## License & credit

This crate is dual-licensed under MIT and Apache-2.0 (your choice).
The bundled ONNX model is licensed by the Silero Team under MIT —
see [`NOTICE`](NOTICE) for the full attribution.

```text
Copyright (c) 2020-present Silero Team   (the model + the Python reference)
Copyright (c) 2026 joe                   (the Rust port)
Licensed under MIT
```

The original Silero VAD project lives at
<https://github.com/snakers4/silero-vad>. If this crate saves you
time, please consider starring the upstream project — they did the
hard work of training the model.

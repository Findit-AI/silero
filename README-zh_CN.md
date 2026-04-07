# silero

[Silero VAD] 的 Rust 实现 — 基于官方 Python 参考项目
[`snakers4/silero-vad`][upstream] 的逐行翻译。

[Silero VAD]: https://github.com/snakers4/silero-vad
[upstream]: https://github.com/snakers4/silero-vad

神经网络本身没有重写 — 我们通过 [`ort`] 加载相同的
`silero_vad.onnx` 模型(~2.3 MB),用 Rust 跑推理。模型**外**的所有
代码(预处理、滚动 LSTM 状态、流式边界事件、padding、段级时间戳
计算)都是上游 Python 参考实现的逐行 port,目的是让相同的音频在
两边产生**逐位等价**的结果。

[`ort`]: https://docs.rs/ort

英文版完整文档: [README.md](README.md)

---

## 为什么要做 Rust 版

上游已经有完整可用的 Python 参考,但在 Rust 应用中接入有几个痛点:

1. **依赖体积**。Python + PyTorch + torchaudio = 5–10 GB。Rust + ORT
   + ONNX 一共大约 30 MB。
2. **吞吐量**。纯 Rust 流水线(音频解码 → 重采样 → VAD → STT)
   不用穿越 Python ↔ Rust FFI 边界,也不用受 GIL 限制。
3. **隐藏的 ONNX 调用约束**。ONNX 模型期望 `[1, 64+512]` 形状的
   音频张量 — 每个 chunk 前面要拼 64 个样本的滚动上下文。直接传
   原始 512 样本会触发模型内部的"too short"分支,**所有输入都返回
   接近零的概率**。这个细节只在 Python `OnnxWrapper` 源码里能找到。
   我们已经踩过坑了,你不用再踩一次。

模型本身和上游完全相同;下面的精度数据在浮点容差范围内匹配 Python
参考实现。

---

## 快速上手

```toml
[dependencies]
silero = { git = "https://github.com/Findit-AI/silero" }
```

```rust
use silero::{speech_timestamps, SampleRate, VadConfig};

let audio: Vec<f32> = read_my_audio();   // 16 kHz mono f32 PCM

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

## 三层 API

按需求挑最低的那一层:

| 层级 | 类型 | 适用场景 |
|---|---|---|
| `speech_timestamps()` | 一次性批量函数 | 整段音频已在内存,需要返回段列表(离线索引、STT 前置过滤) |
| `VadIterator` | 流式状态机,事件驱动 | 实时音频,需要低延迟拿到 Start/End 事件 |
| `SileroModel` | 底层 ONNX 包装 | 需要逐 chunk 看概率、自定义边界逻辑、跨多流共享一个加载好的模型 |

### 批量 API — `speech_timestamps`

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
# Ok::<_, silero::SileroError>(())
```

等价于上游 Python 的 `get_speech_timestamps`,默认参数完全相同
(`threshold=0.5`、`min_speech_duration_ms=250`、
`min_silence_duration_ms=100`、`speech_pad_ms=30`),包括相同的复杂
段间 padding 规则。

### 流式 API — `VadIterator`

```rust
use silero::{Event, VadConfig, VadIterator};

let mut vad = VadIterator::new("models/silero_vad.onnx", VadConfig::default())?;

for chunk in incoming_audio.chunks_exact(vad.chunk_samples()) {
    if let Some(event) = vad.process(chunk)? {
        match event {
            Event::Start(sample) => println!("speech started @ {sample}"),
            Event::End(sample) => println!("speech ended   @ {sample}"),
        }
    }
}

if let Some(Event::End(sample)) = vad.finish() {
    println!("flushed end @ {sample}");
}
# Ok::<_, silero::SileroError>(())
```

流式状态机是上游 Python `VADIterator` 的一比一翻译,只有一个故意
的扩展:`finish()` 会为流末尾还没关的语音段返回一个合成的 `End`
事件(上游 Python 会静默丢弃)。其他行为与参考实现完全一致。

## 默认参数对齐上游

| 参数 | 默认值 | 来源 |
|---|---|---|
| `threshold` | `0.5` | 上游 |
| `negative_threshold` | `max(threshold - 0.15, 0.01)` (lazy) | 上游 |
| `min_speech_duration_ms` | `250` | 上游 |
| `min_silence_duration_ms` | `100` | 上游 |
| `speech_pad_ms` | `30` | 上游 |
| `min_silence_at_max_speech_ms` | `98` | 上游 |
| 采样率 | `16 kHz` | 上游 |

## 采样率支持

Silero VAD 训练于 8 kHz 和 16 kHz。`speech_timestamps` 还接受
16 kHz 的整数倍(32、48、96 kHz...)并自动做朴素抽取(decimation)
降到 16 kHz,跟上游 Python 行为一致。其他采样率(44.1 kHz、22.05
kHz...)需要调用方先重采样。可选的 `resample` feature 启用 `rubato`
高质量重采样器:

```toml
[dependencies]
silero = { git = "...", features = ["resample"] }
```

```rust
let audio_16k = silero::resample::resample(&audio_44k, 44_100, 16_000)?;
```

## 性能

Apple Silicon (M 系列) 单线程,16 kHz 单声道:

| 任务 | 延迟 | 吞吐 |
|---|---|---|
| `SileroModel::process` (单个 32ms chunk) | ~120 µs | 单次 ~270× 实时 |
| `speech_timestamps` (10 秒) | ~37 ms | ~270× 实时 |
| `VadIterator` (10 秒) | ~37 ms | ~270× 实时 |

## 不做的事

- **不重写神经网络**。Conv / LSTM / sigmoid 仍然由 ONNX Runtime
  执行(已经是手工优化过的)。从零写一个不会更快,只会增加维护成本。
- **不检测唱腔**。Silero 的训练目标是"说话"不是"人声"。流行歌、
  rap、咏叹调里的演唱会被故意判为非语音。这对 STT 前置过滤来说是
  正确的行为(否则 whisper 会幻觉出歌词)。如果你需要识别歌曲,
  请用音乐分类模型或 CLAP 类音频文本模型。
- **不做音频解码**。请喂解码后的 f32 PCM。`wav` feature 是有意精简
  的 — 超出范围请用 [`symphonia`] 之类的专门库。

[`symphonia`]: https://github.com/pdeljanov/Symphonia

## 与 Python 参考的对照

| 性质 | 上游 Python | 本 crate |
|---|---|---|
| 神经网络一致 | ✓ | ✓ (相同的 `silero_vad.onnx`) |
| 默认值一致 | ✓ | ✓ |
| 流式 API | `VADIterator` | `VadIterator` |
| 批量 API | `get_speech_timestamps` | `speech_timestamps` |
| Padding 规则 | 文档化 | 一比一 port |
| 最大段长拆分 | 文档化 | 一比一 port |
| 自动抽取 32/48 kHz | ✓ | ✓ |
| 流末尾自动 emit end | ✗ | ✓ 通过 `finish()` |
| 嵌入 ONNX 文件 | ✓ (~2.3 MB) | ✓ (`models/silero_vad.onnx`) |
| 运行依赖 | Python + torch + torchaudio | `ort` + `ndarray` |
| 单二进制部署 | ✗ | ✓ |

## 许可与归属

本 crate 双许可: MIT 或 Apache-2.0(任选)。捆绑的 ONNX 模型由
Silero Team 在 MIT 下发布 — 完整归属见 [`NOTICE`](NOTICE)。

```text
Copyright (c) 2020-present Silero Team   (模型 + Python 参考)
Copyright (c) 2026 joe                   (Rust 翻译)
Licensed under MIT
```

原始 Silero VAD 项目地址:
<https://github.com/snakers4/silero-vad>。如果本 crate 帮你省了
时间,请去给上游点个 star — 训练模型的辛苦活是他们做的。

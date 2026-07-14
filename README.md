# ComfyUI-YogurtNodes-LingBotVideo

[LingBot-Video](https://github.com/Robbyant/lingbot-video) 的 ComfyUI 本地推理节点，支持：

- Dense 1.3B 与 MoE 30B-A3B
- 文生图（T2I）
- 文生视频（T2V）
- 图文生视频（TI2V）
- Qwen3.6-27B + LingBot LoRA Prompt Rewriter
- 24GB 显卡上的 MoE block swap
- 显式下载开关；默认严格使用本地模型

## 安装

将本项目放入 ComfyUI 的 `custom_nodes`：

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-YogurtNodes-LingBotVideo/
```

使用 **ComfyUI 实际运行的 Python 环境** 安装依赖：

```bash
pip install -r custom_nodes/ComfyUI-YogurtNodes-LingBotVideo/requirements.txt
```

依赖使用范围约束，不锁死补丁版本。插件基于 `transformers>=4.57.0,<5` 实现，不要求或支持 Transformers 5.x。

安装或更新依赖后完整重启 ComfyUI。

## 模型放置

插件只注册以下 ComfyUI 模型根目录：

```text
ComfyUI/models/LingBotVideo/
```

不会使用 Hugging Face cache，也不会自动扫描其他磁盘目录。可以复制模型目录，也可以自行使用目录链接或 junction 链接到这里。

推荐完整结构：

```text
ComfyUI/
└── models/
    └── LingBotVideo/
        ├── Robbyant--lingbot-video-dense-1.3b/
        │   ├── transformer/
        │   ├── text_encoder/
        │   ├── processor/
        │   ├── vae/
        │   ├── scheduler/
        │   └── model_index.json
        │
        ├── Robbyant--lingbot-video-moe-30b-a3b/
        │   ├── transformer/
        │   ├── text_encoder/
        │   ├── processor/
        │   ├── vae/
        │   ├── scheduler/
        │   ├── refiner/
        │   └── model_index.json
        │
        ├── Qwen--Qwen3.6-27B/
        │   ├── config.json
        │   ├── processor_config.json
        │   ├── tokenizer.json
        │   └── model*.safetensors
        │
        └── Robbyant--lingbot-video-rewriter-lora/
            ├── adapter_config.json
            └── adapter_model.safetensors
```

注意：

- Dense 与 MoE 必须保留官方发布的完整 Diffusers 目录结构，不能只放 Transformer 权重。
- Rewriter Base 和 Rewriter LoRA 是两个独立目录。
- MoE 目录中的 `refiner/` 可以保留，但当前生成节点运行的是 base T2I/T2V/TI2V pipeline，尚未提供独立 Refiner 节点。
- 文件可能按多个 safetensors shard 分片，保持官方文件名和 index 文件不变。

### 使用现有外部模型目录

例如模型原本位于：

```text
D:\Models\hf\Robbyant--lingbot-video-moe-30b-a3b
```

可将它链接到：

```text
ComfyUI\models\LingBotVideo\Robbyant--lingbot-video-moe-30b-a3b
```

Windows PowerShell 示例（请按实际 ComfyUI 路径修改）：

```powershell
New-Item `
  -ItemType Junction `
  -Path "D:\Codes\AIArt\ComfyUI\models\LingBotVideo\Robbyant--lingbot-video-moe-30b-a3b" `
  -Target "D:\Models\hf\Robbyant--lingbot-video-moe-30b-a3b"
```

Dense、Rewriter Base 和 Rewriter LoRA 可以用同样方式分别建立链接。链接名必须使用上面目录树中的名称。

插件只看到 ComfyUI 目录中的链接入口，不会修改外部模型目录。

## 节点

节点位于：

```text
YogurtLingBotVideo
```

### Load LingBot Video Model

加载 Dense 或 MoE pipeline。

主要参数：

| 参数 | 说明 |
| --- | --- |
| `model_name` | 选择 `models/LingBotVideo` 下的 Dense 或 MoE 模型 |
| `mode` | `t2i`、`t2v` 或 `ti2v`；必须与后续工作流用途一致 |
| `transformer_dtype` | 推荐 `auto` 或 `bfloat16` |
| `cpu_offload` | 默认 `true`；MoE 24GB 显卡必须开启 |
| `moe_gpu_blocks` | MoE 中长期驻留 GPU 的 block 数，Dense 会忽略此参数 |
| `download_model` | 默认 `false`，只有主动开启才下载模型 |

### LingBot Video Generate

执行 T2I、T2V 或 TI2V。输出是 ComfyUI `IMAGE` batch：

- T2I 输出一张图片。
- T2V/TI2V 输出所有视频帧，可连接 ComfyUI 的视频合成或保存节点。
- TI2V 必须连接 `image` 输入。

官方默认参数：

```text
width          832
height         480
steps          40
guidance_scale 3.0
shift          3.0
```

帧数规则：

```text
1 或 4n+1
```

常见视频帧数为 `81`。宽高必须是 16 的倍数。

### Load LingBot Prompt Rewriter

加载：

- `Qwen--Qwen3.6-27B`
- `Robbyant--lingbot-video-rewriter-lora`

Rewriter 很大，会与视频模型争抢显存。建议在 Prompt Rewrite 节点保持“改写后释放显存”开启。

### LingBot Prompt Rewrite

将普通自然语言 Prompt 扩写并转换为 LingBot 推荐的结构化 JSON，同时输出详细的中间 Prompt。

`release_rewriter_after_rewrite=true`（默认）时：

- 两阶段改写完成后释放 Rewriter 显存。
- 改写异常时也会尝试释放，并保留原始异常。
- 同一缓存 handle 下次执行会从本地目录重新按 `device_map=auto` 加载。

连续反复改写、并且暂时不运行视频模型时，可以关闭该选项以减少重复加载时间。

## 基本工作流

仓库中的 `example_workflows/` 提供可直接导入的示例：

- [`lingbot_dense_t2i.json`](example_workflows/lingbot_dense_t2i.json)：Dense 文生图并保存 PNG。
- [`lingbot_dense_t2v.json`](example_workflows/lingbot_dense_t2v.json)：Dense 文生视频并保存 WEBM。
- [`lingbot_dense_ti2v.json`](example_workflows/lingbot_dense_ti2v.json)：参考图加文本生成视频；导入后先选择输入图片。
- [`lingbot_prompt_rewriter_t2v.json`](example_workflows/lingbot_prompt_rewriter_t2v.json)：单独扩写并预览 T2V Prompt，确认后可复制到生成工作流，避免同时驻留两个大模型。

示例默认不下载模型。请先按“模型放置”准备对应目录，或明确需要下载时再手动打开 Loader 的 `download_model`。

### T2I 文生图

```text
Load LingBot Video Model (mode=t2i)
    → LingBot Video Generate
    → Save Image
```

T2I 会强制生成一帧，Generate 节点上的 `num_frames` 不影响结果。

### T2V 文生视频

```text
Load LingBot Video Model (mode=t2v)
    → LingBot Video Generate
    → Video Combine / Save Video
```

### TI2V 图文生视频

```text
Load Image ───────────────────────────────┐
                                          ↓
Load LingBot Video Model (mode=ti2v) → LingBot Video Generate
                                          ↓
                               Video Combine / Save Video
```

### 使用 Prompt Rewriter

```text
Load LingBot Prompt Rewriter
    → LingBot Prompt Rewrite
    → prompt_json
    → LingBot Video Generate.prompt
```

TI2V Prompt Rewrite 也需要连接同一张参考图片。

## Prompt 格式

推荐直接使用 Prompt Rewrite 输出的 JSON。

生成节点也接受普通文本，例如：

```text
A clear glass bottle filled with water sits on a wooden table in warm sunlight.
```

普通文本会自动包装为：

```json
{"comprehensive_description":"A clear glass bottle filled with water sits on a wooden table in warm sunlight."}
```

这只能保证输入格式有效，不等价于官方 Rewriter 的详细扩写。过短或缺乏镜头、主体、材质、光线和动作描述的 Prompt，生成质量通常较差。

`negative_prompt` 留空时，插件会根据 T2I 或视频模式使用官方默认 negative prompt。

## 显存与内存

### Dense 1.3B

24GB RTX 4090 可以运行 Dense T2I。一般建议：

```text
transformer_dtype = bfloat16 或 auto
cpu_offload       = true
```

显存充足并希望减少模型搬运时，可以关闭 `cpu_offload`。实际峰值会随模式、分辨率、帧数、ComfyUI 中同时加载的其他模型以及 PyTorch 后端变化。

### MoE 30B-A3B

MoE Transformer 的 BF16 权重约 60.27GB，不能完整放入 24GB 显存。插件使用混合 block swap：

- 前 `moe_gpu_blocks` 个 block 长期驻留 GPU。
- 其余 block 在执行前从 CPU 上载，执行后卸载。
- 非 block 的小型模块驻留 GPU。
- 不改变 MoE routing、expert、attention、scheduler 或采样数学。

RTX 4090 24GB 实测建议：

| `moe_gpu_blocks` | 适用情况 | 说明 |
| ---: | --- | --- |
| `0` | 最保守 | 全部 block swap，显存最低，速度最慢 |
| `8`–`10` | 需要给其他节点留显存 | 更安全，但每步搬运更多权重 |
| `12` | **4090 推荐默认值** | 实测约 19.4GB，剩余约 4.8GB |
| `14`–`16` | 激进调优 | 可能更快，但分辨率或其他模型占用稍高就可能 OOM |
| `48` | 不适用于 24GB | 等同所有 block 常驻，4090 无法容纳 |

默认 `12` 的本机参考结果（RTX 4090、832×480、T2I、40 steps）：

```text
稳定显存占用约 19.4GB
40 步采样约 6 分 33 秒
```

这些是单机实测，不是所有系统的固定性能保证。PCIe 速度、系统内存、分页文件、后台GPU占用和Torch版本都会影响耗时。

系统内存建议：

- 64GB 可以运行，但接近模型体积边缘，错误的全量offload策略可能触发分页并严重变慢。
- 更推荐 96GB 或以上，为模型权重、文本编码器、VAE和ComfyUI保留余量。
- 首次Loader需要读取并布置约60GB权重，耗时可能数分钟。
- ComfyUI复用同一个Loader输出时，后续生成无需每次重新加载全部模型。

如果发生 OOM，依次尝试：

1. 降低 `moe_gpu_blocks`。
2. 确认 `cpu_offload=true`。
3. 关闭或释放 Prompt Rewriter、其他模型和不用的工作流分支。
4. 降低分辨率或视频帧数。
5. 重启 ComfyUI，清除其他工作流遗留的模型占用。

## 模型下载行为

`download_model=false` 是默认值：

- 不访问网络。
- 不自动下载。
- 模型缺失或目录无效时直接报错。

只有显式设置 `download_model=true` 时，插件才下载所选官方模型，并直接写入：

```text
ComfyUI/models/LingBotVideo/
```

下载不经过 Hugging Face cache。没有后台更新检查、遥测或自动联网行为。

MoE 与 Qwen Rewriter Base 都非常大，开启下载前请确认磁盘空间。

## 已验证范围

- Dense 1.3B T2I：与官方参考在相同 Prompt、seed 和参数下逐像素一致。
- MoE 30B-A3B T2I：与官方参考在相同 Prompt、seed 和参数下逐像素一致。
- MoE block swap 只改变权重驻留位置；实测输出与官方参考逐像素一致。
- 自动化测试：67 项通过。

T2V、TI2V 和 Refiner 尚未完成同等级的逐帧官方 A/B 验证，因此不宣称这些模式已经达到逐帧、逐像素一致。

## 常见问题

### 模型下拉框为空

检查目录是否位于 `ComfyUI/models/LingBotVideo`，并确认完整的 `model_index.json`、`transformer`、`text_encoder`、`processor`、`vae` 和 `scheduler` 组件存在。放置或链接模型后重启 ComfyUI。

### 普通Prompt生成结果很差

LingBot 更适合详细、结构化的视觉描述。使用 Prompt Rewriter，或手工补充主体、环境、构图、镜头、光线、材质和动作。

### MoE加载后很久没有开始采样

首次加载需要读取约60GB Transformer，并建立GPU常驻block和CPU swap block。观察系统内存、磁盘活动和ComfyUI控制台；后续复用同一Loader会快很多。

### MoE运行时只有几GB显存但特别慢

检查是否使用了新版本节点，并将 `moe_gpu_blocks` 设为 `12` 左右。`0` 会将所有block换入换出，显存低但速度明显更慢。

### 修改 `moe_gpu_blocks` 后没有变化

Loader节点可能被ComfyUI缓存。修改参数并重新排队；如果仍使用旧handle，重启ComfyUI后再运行。

## 上游许可

内嵌的 LingBot pipeline 源码派生自 Apache-2.0 上游项目：

```text
yogurt_lingbot_video/upstream/LICENSE.upstream
```

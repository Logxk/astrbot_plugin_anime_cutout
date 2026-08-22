# 动漫抠图 (astrbot_plugin_anime_cutout)

基于 [anime-segmentation](https://github.com/SkyTNT/anime-segmentation)（ISNet / ISNetDIS）的
AstrBot 动漫人物一键抠图插件。发送一张动漫图片即可去除背景，得到透明底 PNG，
也可以输出白底、遮罩或对比图。

## 功能

- 一张图一键去背景，输出透明背景 PNG
- 支持输出模式：`transparent`（透明背景）/ `white`（白底）/ `mask`（遮罩）/ `compare`（对比图）
- 支持直接发送图片、回复带图片的消息、附带图片链接三种触发方式
- 模型惰性加载、结果缓存，多平台通用
- 配置面板可视化设置模型路径 / 推理设备 / 输入尺寸 / 混合精度等

## 使用

| 指令 | 说明 |
| --- | --- |
| `图片 + /cutout` | 透明背景抠图 |
| `/cutout white` | 输出白底图 |
| `/cutout mask` | 输出遮罩（黑白蒙版） |
| `/cutout compare` | 输出「原图｜抠图｜遮罩」对比图 |
| `/cutout reload` | 重新加载模型 |
| `/cutout_help` | 查看使用说明与当前配置 |

支持中文别名：`/抠图`、`/去背景`、`/去底`。

## 安装

1. 确认 AstrBot 运行环境中已安装 `torch` / `torchvision`（如需 CUDA 加速，请手动安装 CUDA 版 PyTorch）
2. 将本插件目录放入 `AstrBot/data/plugins/astrbot_plugin_anime_cutout`
3. 在 AstrBot WebUI 的插件管理中启用并重载插件，插件会自动安装其余依赖
4. 下载抠图模型 `isnetis.ckpt`（约 200MB），放入插件目录 `models/` 下，
   或在配置项 `model_path` 中指定其路径。下载地址见下方 [模型](#模型) 章节

## 模型

插件模型为官方 `isnetis.ckpt`（约 200MB），**不随仓库分发，需自行下载**。

### 下载地址

| 来源 | 链接 |
| --- | --- |
| HuggingFace 官方站 | https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.ckpt |
| HuggingFace 镜像站（国内加速） | https://hf-mirror.com/skytnt/anime-seg/resolve/main/isnetis.ckpt |
| 模型仓库页面 | https://huggingface.co/skytnt/anime-seg |

> 镜像站与官方站文件一致，国内网络访问官方站缓慢或失败时请改用镜像站链接。

### 下载后放置

将下载到的 `isnetis.ckpt` 放到以下任一位置，插件会按优先级查找：

1. 插件配置 `model_path` 中指定的路径
2. 数据目录 `data/plugin_data/astrbot_plugin_anime_cutout/isnetis.ckpt`
3. 插件目录 `models/isnetis.ckpt`（兼容 `model/` 目录名）

> 注意：AstrBot 实际加载的是 **已安装副本**
> `data/plugins/astrbot_plugin_anime_cutout/`，请将模型放在该目录下
> （而非开发工作区），否则插件找不到。

### 自动下载

若开启配置项 `auto_download_model`，且上述位置均未找到模型，插件会自动从
HuggingFace `skytnt/anime-seg` 下载到
`data/plugin_data/astrbot_plugin_anime_cutout/isnetis.ckpt`。

国内网络下，自动下载默认已走镜像站加速（配置项 `hf_use_mirror` 默认开启，
端点为 `https://hf-mirror.com`），无需手动设置环境变量。

如需自定义端点，可修改配置项 `hf_mirror_url`；海外服务器可关闭
`hf_use_mirror` 直连官方站。也可在启动 AstrBot 前设置环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

（Windows PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。

## 配置项

### 模型与推理

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `model_path` | 空 | 自定义模型权重路径 |
| `auto_download_model` | true | 找不到模型时自动下载 |
| `hf_repo` | skytnt/anime-seg | 自动下载仓库 |
| `hf_file` | isnetis.ckpt | 自动下载文件名 |
| `hf_use_mirror` | true | 自动下载走 HF 镜像站（国内建议开启） |
| `hf_mirror_url` | https://hf-mirror.com | 镜像站端点 |
| `device` | auto | 推理设备：auto / cpu / cuda:0 |
| `img_size` | 1024 | 网络输入尺寸，CPU 建议 640 |
| `use_amp` | true | GPU 混合精度 |
| `output_mode` | transparent | 默认输出模式 |

### 安全与缓存

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `max_image_pixels` | 20000000 | 输入图片最大像素数（宽×高），防范解压炸弹 |
| `max_image_file_size_mb` | 50 | 输入图片文件大小上限（MB） |
| `cache_enabled` | true | 启用结果缓存（按图片内容哈希+模式） |
| `cache_max_entries` | 50 | 缓存条目上限，超出后淘汰最旧条目 |
| `cache_ttl_hours` | 24 | 缓存输出文件保留时长（小时） |
| `cleanup_interval_minutes` | 30 | 自动清理任务执行间隔（分钟） |

## 安全机制

- **内存安全快照**：每次处理前将输入图片一次性读入内存（带文件大小校验），以不可变的字节快照作为后续校验与推理的唯一数据源，从源头消除 TOCTOU 风险，无需磁盘临时副本。
- **解压炸弹防护**：通过 PIL 仅解析图片头部获取分辨率（不解码像素），超过 `max_image_pixels` 的图片将被拒绝；cv2 解码后进行二次校验，双重保障。
- **文件大小限制**：超过 `max_image_file_size_mb` 的图片将被拒绝。
- **结果缓存**：按图片内容 blake2b 哈希 + 输出模式缓存抠图结果，相同图片重复发送时直接返回缓存，跳过推理。缓存采用 LRU 策略（线程安全），超出 `cache_max_entries` 后自动淘汰最旧条目并删除其文件。
- **自动清理**：后台任务每 `cleanup_interval_minutes` 分钟执行一次，删除超过 `cache_ttl_hours` 的缓存输出文件与残留临时文件，避免磁盘空间无限增长。插件卸载时优雅停止清理任务并执行最终清理。
- **模型引用捕获**：处理前捕获模型引用，即使 `/cutout reload` 重置模型，正在处理的请求仍使用有效的模型对象，避免竞态。
- **原子写入**：结果图片先写入临时文件再原子重命名，跨设备时回退 `shutil.move`。

## 兼容平台

支持所有支持收发图片的 AstrBot 平台适配器（aiocqhttp / qq_official / telegram /
discord / lark / dingtalk / wecom / satori / kook / line / matrix / mattermost 等）。

## 感谢

模型与推理代码来自 [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation)
（MIT License），模型权重 `isnetis.ckpt` 来自该仓库在 HuggingFace 的发布
（[skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg)）。
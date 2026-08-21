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
4. 自行安装抠图模型

## 模型

插件模型为官方 `isnetis.ckpt`（约 200MB）。获取方式按优先级：

1. 插件配置 `model_path` 中指定的路径
2. 插件目录 `models/isnetis.ckpt`（本插件已内置）
3. 开启 `auto_download_model` 时，自动从 HuggingFace `skytnt/anime-seg` 下载到
   `data/plugin_data/astrbot_plugin_anime_cutout/isnetis.ckpt`

## 配置项

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `model_path` | 空 | 自定义模型权重路径 |
| `auto_download_model` | true | 找不到模型时自动下载 |
| `hf_repo` | skytnt/anime-seg | 自动下载仓库 |
| `hf_file` | isnetis.ckpt | 自动下载文件名 |
| `device` | auto | 推理设备：auto / cpu / cuda:0 |
| `img_size` | 1024 | 网络输入尺寸，CPU 建议 640 |
| `use_amp` | true | GPU 混合精度 |
| `output_mode` | transparent | 默认输出模式 |

## 兼容平台

支持所有支持收发图片的 AstrBot 平台适配器（aiocqhttp / qq_official / telegram /
discord / lark / dingtalk / wecom / satori / kook / line / matrix / mattermost 等）。

## 感谢

模型与推理代码来自 [SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation)
（MIT License，随插件附带 `models/isnetis.ckpt` 亦来自该仓库的 HuggingFace 发布）。
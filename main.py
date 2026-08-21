import asyncio
import os
import re
import sys
import time
import uuid
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

PLUGIN_DIR = Path(__file__).resolve().parent

# AstrBot 通过包路径导入插件，不会把插件目录加入 sys.path，
# 运行时需手动添加，否则 `import anime_cutout_lib` 找不到子包。
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

# 模型加载互斥锁，避免并发触发重复加载
_MODEL_LOCK = asyncio.Lock()

MODE_ALIASES = {
    "transparent": "transparent",
    "alpha": "transparent",
    "png": "transparent",
    "透明": "transparent",
    "white": "white",
    "白底": "white",
    "白": "white",
    "mask": "mask",
    "maskonly": "mask",
    "遮罩": "mask",
    "compare": "compare",
    "comp": "compare",
    "对比": "compare",
    "原图": "compare",
}

OUTPUT_EXT = {
    "transparent": ".png",
    "white": ".jpg",
    "mask": ".jpg",
    "compare": ".jpg",
}


class AnimeCutoutPlugin(Star):
    """动漫图片一键抠图插件，基于 ISNet（isnetis.ckpt）。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 模型状态（首次调用时惰性加载）
        self.model = None
        self.model_path = None
        self.model_device = None
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_anime_cutout")

    def _parse_mode(self, mode, message_str: str) -> str:
        """从命令参数 / 消息文本解析输出模式，未命中则使用配置默认值。"""
        candidates = []
        if mode:
            candidates.append(str(mode))
        for tok in str(message_str or "").split():
            t = str(tok).strip().lstrip("/").lower()
            candidates.append(t)
        for c in candidates:
            if c.lower() in MODE_ALIASES:
                return MODE_ALIASES[c.lower()]
        return str(self.config.get("output_mode", "transparent"))

    def _resolve_device(self) -> str:
        dev = str(self.config.get("device", "auto"))
        if dev == "auto":
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return dev

    def _download_model_sync(self) -> str:
        """从 HuggingFace 下载模型到插件数据目录。"""
        dest = self.data_dir / "isnetis.ckpt"
        if dest.exists():
            return str(dest)
        repo = str(self.config.get("hf_repo", "skytnt/anime-seg"))
        filename = str(self.config.get("hf_file", "isnetis.ckpt"))
        logger.info(f"[cutout] 开始从 HuggingFace({repo}) 下载模型 ...")
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=repo, filename=filename, local_dir=self.data_dir)
        logger.info(f"[cutout] 模型下载完成: {dest}")
        return str(dest)

    def _resolve_model_path(self) -> str:
        cfg_path = str(self.config.get("model_path", "") or "").strip()
        if cfg_path and os.path.exists(cfg_path):
            return cfg_path
        bundled = PLUGIN_DIR / "models" / "isnetis.ckpt"
        if bundled.exists():
            return str(bundled)
        if self.config.get("auto_download_model", True):
            return self._download_model_sync()
        return ""

    def _load_model_sync(self, path: str, device: str):
        """在线程中加载模型，避免阻塞事件循环。"""
        from anime_cutout_lib import inference as seg

        t0 = time.time()
        model = seg.load_model(path, device)
        logger.info(f"[cutout] 模型加载完成, device={device}, 耗时 {time.time() - t0:.1f}s")
        return model

    async def ensure_model(self):
        """惰性加载模型（带互斥与缓存）。"""
        path = self._resolve_model_path()
        device = self._resolve_device()
        if self.model is not None and self.model_path == path and self.model_device == device:
            return
        if not path:
            raise RuntimeError(
                "未找到模型权重文件。请在插件配置中设置 model_path，"
                "或将 isnetis.ckpt 放入插件目录 models/ 下，或开启 auto_download_model。"
            )
        async with _MODEL_LOCK:
            if self.model is not None and self.model_path == path and self.model_device == device:
                return
            self.model = await asyncio.to_thread(self._load_model_sync, path, device)
            self.model_path = path
            self.model_device = device

    def _process_sync(self, img_path: str, mode: str) -> str:
        """在线程中完成 读取 -> 抠图 -> 渲染 -> 保存，返回输出文件路径。"""
        import cv2

        from anime_cutout_lib import inference as seg

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"无法读取图片: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_size = int(self.config.get("img_size", 1024))
        use_amp = bool(self.config.get("use_amp", True))
        mask = seg.get_mask(self.model, rgb, use_amp=use_amp, s=img_size)
        result = seg.render_result(bgr, mask, mode)

        ext = OUTPUT_EXT.get(mode, ".png")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.data_dir / f"cutout_{uuid.uuid4().hex}{ext}"
        ok, buf = cv2.imencode(ext, result)
        if not ok:
            raise ValueError("结果图片编码失败")
        buf.tofile(str(out_path))
        return str(out_path)

    async def _find_image(self, event: AstrMessageEvent):
        """从消息链 / 回复消息 / 文本 URL 中找到一个可用的图片本地路径。"""
        comps = event.get_messages()

        # 1) 消息中的图片
        for comp in comps:
            if isinstance(comp, Comp.Image):
                try:
                    return await comp.convert_to_file_path()
                except Exception as e:
                    logger.warning(f"[cutout] 图片解析失败: {e}")
        # 2) 回复消息中的图片
        for comp in comps:
            if isinstance(comp, Comp.Reply):
                for sub in (getattr(comp, "chain", None) or []):
                    if isinstance(sub, Comp.Image):
                        try:
                            return await sub.convert_to_file_path()
                        except Exception as e:
                            logger.warning(f"[cutout] 回复图片解析失败: {e}")
        # 3) 消息文本中的图片 URL
        for url in re.findall(r"https?://\S+", str(event.message_str or "")):
            if re.search(r"\.(png|jpe?g|webp|gif|bmp)(\?\S*)?$", url, re.IGNORECASE):
                try:
                    return await Comp.Image.fromURL(url).convert_to_file_path()
                except Exception as e:
                    logger.warning(f"[cutout] 图片 URL 下载失败: {e}")
        return None

    @filter.command("cutout", alias={"抠图", "去背景", "去底"})
    async def cutout(self, event: AstrMessageEvent, mode: str = ""):
        '''
        动漫图片一键抠图。发送图片并附带 /cutout 即可去除背景。
        可指定输出模式：/cutout white（白底）、/cutout mask（遮罩）、/cutout compare（对比图），默认透明背景。
        支持回复带图片的消息，也支持附带图片链接。发送 /cutout reload 可重新加载模型。
        '''
        if str(mode).strip().lower() == "reload":
            self.model = None
            self.model_path = None
            self.model_device = None
            yield event.plain_result("模型已重置，将在下次使用时重新加载。")
            return

        t0 = time.time()
        yield event.plain_result("正在处理…请稍候。")
        mode = self._parse_mode(mode, event.message_str)

        img_path = await self._find_image(event)
        if not img_path:
            yield event.plain_result(
                "未找到图片。请发送「图片 + /cutout」或回复一条带图片的消息。"
            )
            return

        try:
            await self.ensure_model()
            out_path = await asyncio.to_thread(self._process_sync, img_path, mode)
        except Exception as e:
            logger.error(f"[cutout] 处理失败: {e}")
            yield event.plain_result(f"处理失败：{e}")
            return

        yield event.image_result(out_path)
        logger.info(f"[cutout] 完成，耗时 {time.time() - t0:.1f}s, mode={mode}")

    @filter.command("cutout_help")
    async def cutout_help(self, event: AstrMessageEvent):
        '''查看动漫抠图插件的使用说明与当前配置。'''
        cfg = {
            "model_path": str(self.config.get("model_path", "")) or "(自动)",
            "当前设备": self.model_device or "(未加载)",
            "img_size": self.config.get("img_size", 1024),
            "默认输出模式": self.config.get("output_mode", "transparent"),
        }
        lines = [
            "动漫抠图 使用说明：",
            "· 发送「图片 + /cutout」→ 透明背景抠图",
            "· /cutout white → 白底 | /cutout mask → 遮罩 | /cutout compare → 对比图",
            "· 可回复带图片的消息触发",
            "· /cutout reload 重新加载模型",
            "",
            "当前配置：",
        ]
        lines.extend(f"{k}: {v}" for k, v in cfg.items())
        yield event.plain_result("\n".join(lines))

    async def terminate(self):
        """插件卸载 / 停用时释放模型显存。"""
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
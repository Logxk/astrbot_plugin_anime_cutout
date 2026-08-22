"""动漫抠图插件 — 基于 ISNet 的动漫人物一键抠图。

架构：
  PluginConfig      — 类型安全的配置访问（带边界校验）
  ModelManager      — 模型加载、设备选择与状态管理
  CacheManager      — LRU 结果缓存与磁盘清理（线程安全）
  ImageProcessor    — 图片校验、解码、推理与原子保存
  MessageExtractor  — 从消息事件中提取图片路径
  AnimeCutoutPlugin — 插件入口，仅负责编排上述组件
"""

import asyncio
import hashlib
import io
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

# ------------------------------------------------------------------ #
#  常量
# ------------------------------------------------------------------ #

PLUGIN_DIR = Path(__file__).resolve().parent

# AstrBot 通过包路径导入插件，不会把插件目录加入 sys.path，
# 运行时需手动添加，否则 `import anime_cutout_lib` 找不到子包。
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

# 模型加载互斥锁，避免并发触发重复加载
_MODEL_LOCK = asyncio.Lock()

_MODE_ALIASES: dict[str, str] = {
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

# 输出模式 → 文件扩展名（单一真相源）
_OUTPUT_EXT: dict[str, str] = {
    "transparent": ".png",
    "white": ".jpg",
    "mask": ".jpg",
    "compare": ".jpg",
}

# 安全与缓存默认值
_DEFAULT_MAX_PIXELS = 20_000_000
_DEFAULT_MAX_FILE_MB = 50
_DEFAULT_CACHE_ENTRIES = 50
_DEFAULT_CACHE_TTL_HOURS = 24
_DEFAULT_CLEANUP_INTERVAL_MIN = 30
_TMP_RETENTION_SECONDS = 3600
_MIN_MODEL_SIZE = 1_000_000  # 模型文件最小合理大小（字节）


def _safe_remove(path: str) -> None:
    """安全删除文件，忽略不存在的情况，权限错误记 warning。"""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"[cutout] 删除文件失败 {path}: {e}")


def _cache_key(data: bytes, mode: str) -> str:
    """根据图片内容哈希与输出模式生成缓存键。

    使用 blake2b（比 sha256 快约 3 倍）生成 128 位摘要，
    对缓存键而言碰撞概率可忽略。
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(data)
    h.update(mode.encode("utf-8"))
    return h.hexdigest()


def _looks_like_image_url(url: str) -> bool:
    """判断 URL 是否指向图片（按扩展名粗筛）。"""
    return bool(
        re.search(
            r"\.(png|jpe?g|webp|gif|bmp)(\?\S*)?$",
            url,
            re.IGNORECASE,
        )
    )


# ------------------------------------------------------------------ #
#  PluginConfig — 类型安全的配置访问
# ------------------------------------------------------------------ #


class PluginConfig:
    """类型安全的配置访问，带默认值与边界校验。

    每次访问均从底层 AstrBotConfig 实时读取，
    配置变更无需重载即可生效。
    """

    def __init__(self, config: AstrBotConfig) -> None:
        self._cfg = config

    @staticmethod
    def _int(
        cfg: AstrBotConfig,
        key: str,
        default: int,
        lo: Optional[int] = None,
        hi: Optional[int] = None,
    ) -> int:
        """读取整数配置项，带类型转换与边界钳制。"""
        try:
            val = int(cfg.get(key, default))
        except (TypeError, ValueError):
            val = default
        if lo is not None and val < lo:
            val = lo
        if hi is not None and val > hi:
            val = hi
        return val

    @staticmethod
    def _bool(cfg: AstrBotConfig, key: str, default: bool) -> bool:
        return bool(cfg.get(key, default))

    @staticmethod
    def _str(
        cfg: AstrBotConfig, key: str, default: str
    ) -> str:
        return str(cfg.get(key, default) or default)

    @property
    def model_path(self) -> str:
        return self._str(self._cfg, "model_path", "").strip()

    @property
    def auto_download_model(self) -> bool:
        return self._bool(
            self._cfg, "auto_download_model", True
        )

    @property
    def hf_repo(self) -> str:
        return self._str(self._cfg, "hf_repo", "skytnt/anime-seg")

    @property
    def hf_file(self) -> str:
        return self._str(self._cfg, "hf_file", "isnetis.ckpt")

    @property
    def hf_use_mirror(self) -> bool:
        """是否使用 HuggingFace 镜像站（国内网络建议开启）。"""
        return self._bool(self._cfg, "hf_use_mirror", True)

    @property
    def hf_mirror_url(self) -> str:
        """HuggingFace 镜像站端点。"""
        return self._str(
            self._cfg, "hf_mirror_url", "https://hf-mirror.com"
        )

    @property
    def device(self) -> str:
        return self._str(self._cfg, "device", "auto")

    @property
    def img_size(self) -> int:
        """网络输入尺寸，钳制到 [64, 4096]。"""
        return self._int(
            self._cfg, "img_size", 1024, lo=64, hi=4096
        )

    @property
    def use_amp(self) -> bool:
        return self._bool(self._cfg, "use_amp", True)

    @property
    def output_mode(self) -> str:
        return self._str(
            self._cfg, "output_mode", "transparent"
        )

    @property
    def max_image_pixels(self) -> int:
        return self._int(
            self._cfg,
            "max_image_pixels",
            _DEFAULT_MAX_PIXELS,
            lo=1,
            hi=200_000_000,
        )

    @property
    def max_image_file_size_mb(self) -> int:
        return self._int(
            self._cfg,
            "max_image_file_size_mb",
            _DEFAULT_MAX_FILE_MB,
            lo=1,
            hi=2048,
        )

    @property
    def cache_enabled(self) -> bool:
        return self._bool(self._cfg, "cache_enabled", True)

    @property
    def cache_max_entries(self) -> int:
        return self._int(
            self._cfg,
            "cache_max_entries",
            _DEFAULT_CACHE_ENTRIES,
            lo=1,
            hi=10000,
        )

    @property
    def cache_ttl_hours(self) -> int:
        return self._int(
            self._cfg,
            "cache_ttl_hours",
            _DEFAULT_CACHE_TTL_HOURS,
            lo=1,
            hi=720,
        )

    @property
    def cleanup_interval_minutes(self) -> int:
        return self._int(
            self._cfg,
            "cleanup_interval_minutes",
            _DEFAULT_CLEANUP_INTERVAL_MIN,
            lo=1,
            hi=1440,
        )


# ------------------------------------------------------------------ #
#  ModelManager — 模型加载、设备选择与状态管理
# ------------------------------------------------------------------ #


class ModelManager:
    """模型加载、设备选择与状态管理。

    职责：
    - 解析模型路径（配置 → 内置 → 自动下载）
    - 惰性加载模型（首次调用时，带互斥锁）
    - 管理 model / model_path / model_device 状态
    - reload 时重置状态，释放显存
    """

    def __init__(
        self, config: PluginConfig, data_dir: Path
    ) -> None:
        self._config = config
        self._data_dir = data_dir
        self.model: object = None
        self.model_path: Optional[str] = None
        self.model_device: Optional[str] = None

    def _resolve_device(self) -> str:
        """解析推理设备，auto 时自动检测 CUDA。"""
        dev = self._config.device
        if dev == "auto":
            import torch  # 惰性：torch 导入耗时数秒
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return dev

    def _download_model_sync(self) -> str:
        """从 HuggingFace 下载模型到插件数据目录。

        Raises:
            RuntimeError: 下载失败或文件不完整。
        """
        dest = self._data_dir / "isnetis.ckpt"
        if dest.exists() and dest.stat().st_size > _MIN_MODEL_SIZE:
            return str(dest)
        repo = self._config.hf_repo
        filename = self._config.hf_file

        # 国内网络直连 huggingface.co 会超时，
        # 默认切换到镜像站（可配置关闭）。
        if self._config.hf_use_mirror:
            os.environ["HF_ENDPOINT"] = (
                self._config.hf_mirror_url
            )

        endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
        logger.info(
            f"[cutout] 开始从 HuggingFace({repo}) 下载模型，"
            f"端点: {endpoint} …"
        )
        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=self._data_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"模型自动下载失败（{e.__class__.__name__}）。"
                f"国内网络请开启「使用 HF 镜像站」配置项"
                f"（默认已开启），或在 WebUI 中将模型文件手动"
                f"放置到插件数据目录后设置 model_path。"
            ) from e
        # 完整性校验：下载后检查文件大小
        if (
            not dest.exists()
            or dest.stat().st_size < _MIN_MODEL_SIZE
        ):
            raise RuntimeError(
                "模型下载失败或文件不完整，"
                "请检查网络后重试或手动下载。"
            )
        logger.info(f"[cutout] 模型下载完成: {dest}")
        return str(dest)

    def _resolve_model_path(self) -> str:
        """按优先级解析模型路径：配置 → 内置 → 自动下载。"""
        cfg_path = self._config.model_path
        if cfg_path and os.path.exists(cfg_path):
            return cfg_path
        bundled = PLUGIN_DIR / "models" / "isnetis.ckpt"
        if bundled.exists():
            return str(bundled)
        if self._config.auto_download_model:
            return self._download_model_sync()
        return ""

    def _load_model_sync(self, path: str, device: str):
        """在线程中加载模型（同步，由 to_thread 调用）。"""
        from anime_cutout_lib import inference as seg

        t0 = time.time()
        model = seg.load_model(path, device)
        logger.info(
            f"[cutout] 模型加载完成, device={device}, "
            f"耗时 {time.time() - t0:.1f}s"
        )
        return model

    async def ensure_model(self) -> object:
        """惰性加载模型（带互斥与缓存），返回当前模型。

        调用方应捕获返回值作为本地引用，防止 reload 后
        使用已被置 None 的 self.model。
        """
        # 注意：_resolve_model_path 可能触发网络下载，
        # _resolve_device 首次会 import torch（耗时数秒），
        # 均必须放入线程执行，否则会阻塞事件循环。
        path = await asyncio.to_thread(
            self._resolve_model_path
        )
        device = await asyncio.to_thread(self._resolve_device)
        if (
            self.model is not None
            and self.model_path == path
            and self.model_device == device
        ):
            return self.model
        if not path:
            raise RuntimeError(
                "未找到模型权重文件。请在插件配置中设置"
                " model_path，或将 isnetis.ckpt 放入插件"
                "目录 models/ 下，或开启 auto_download_model。"
            )
        async with _MODEL_LOCK:
            if (
                self.model is not None
                and self.model_path == path
                and self.model_device == device
            ):
                return self.model
            self.model = await asyncio.to_thread(
                self._load_model_sync, path, device
            )
            self.model_path = path
            self.model_device = device
        return self.model

    def reset(self) -> None:
        """重置模型状态（reload 时调用）。"""
        self.model = None
        self.model_path = None
        self.model_device = None

    async def release(self) -> None:
        """释放模型显存。"""
        self.model = None
        self.model_path = None
        self.model_device = None
        try:
            import torch  # 惰性：与 _resolve_device 一致

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ------------------------------------------------------------------ #
#  CacheManager — LRU 结果缓存与磁盘清理（线程安全）
# ------------------------------------------------------------------ #


class CacheManager:
    """LRU 结果缓存与磁盘清理。

    线程安全：所有 _cache 操作（含迭代）均通过 threading.Lock
    保护，因为 _process_sync 与 _cleanup_sync 均在 to_thread
    线程中执行，可能并发访问同一 OrderedDict。
    """

    def __init__(
        self,
        config: PluginConfig,
        cache_dir: Path,
        tmp_dir: Path,
    ) -> None:
        self._config = config
        self._cache_dir = cache_dir
        self._tmp_dir = tmp_dir
        self._cache: "OrderedDict[str, tuple[str, float]]" = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        """查询缓存，命中且文件存在则返回路径并更新 LRU。"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            out_path, _ts = entry
            if not os.path.exists(out_path):
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return out_path

    def put(self, key: str, out_path: str) -> None:
        """写入缓存并按上限淘汰最旧条目（同时删除其文件）。"""
        with self._lock:
            self._cache[key] = (out_path, time.time())
            self._cache.move_to_end(key)
            max_entries = self._config.cache_max_entries
            while len(self._cache) > max_entries:
                _k, (old_path, _t) = self._cache.popitem(
                    last=False
                )
                _safe_remove(old_path)

    def cleanup_sync(self) -> None:
        """清理过期缓存输出与残留临时文件。

        内存缓存的迭代在 _lock 内执行，避免与 put/get 竞态。
        """
        now = time.time()
        retention = self._config.cache_ttl_hours * 3600
        # 1) 清理过期缓存文件
        self._cleanup_dir(self._cache_dir, retention)
        # 2) 清理过期临时文件（向后兼容旧版本残留）
        self._cleanup_dir(self._tmp_dir, _TMP_RETENTION_SECONDS)
        # 3) 同步内存缓存（锁内安全迭代）
        with self._lock:
            stale = [
                k
                for k, (p, _t) in self._cache.items()
                if not os.path.exists(p)
            ]
            for k in stale:
                self._cache.pop(k, None)

    def _cleanup_dir(
        self, directory: Path, retention: float
    ) -> None:
        """删除目录中超过保留时长的文件。"""
        now = time.time()
        try:
            for f in directory.iterdir():
                if not f.is_file():
                    continue
                if now - f.stat().st_mtime > retention:
                    _safe_remove(str(f))
        except OSError as e:
            logger.warning(
                f"[cutout] 清理目录 {directory} 失败: {e}"
            )

    def clear_tmp(self) -> None:
        """清理临时目录中的所有文件。"""
        try:
            for f in self._tmp_dir.iterdir():
                if f.is_file():
                    _safe_remove(str(f))
        except OSError as e:
            logger.warning(f"[cutout] 清理临时目录失败: {e}")

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def cache_size(self) -> int:
        """当前缓存条目数（GIL 保护下原子读取）。"""
        return len(self._cache)

    @property
    def max_entries(self) -> int:
        return self._config.cache_max_entries


# ------------------------------------------------------------------ #
#  ImageProcessor — 图片校验、解码、推理与原子保存
# ------------------------------------------------------------------ #


class ImageProcessor:
    """图片校验、解码、推理与原子保存。

    不持有模型状态——model 由调用方传入（捕获引用），
    避免模型 reload 后使用失效对象。
    """

    def __init__(
        self,
        config: PluginConfig,
        cache: CacheManager,
        output_dir: Path,
    ) -> None:
        self._config = config
        self._cache = cache
        self._output_dir = output_dir

    def read_image_bytes(self, path: str) -> bytes:
        """读取图片到内存并校验文件大小。

        一次性读入内存即为安全快照，无需磁盘临时副本，
        从源头消除 TOCTOU 风险。

        Raises:
            ValueError: 文件不存在、为空或超过大小上限。
        """
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"图片文件不存在: {path}")
        size = p.stat().st_size
        if size <= 0:
            raise ValueError("图片文件为空。")
        max_mb = self._config.max_image_file_size_mb
        if size > max_mb * 1024 * 1024:
            raise ValueError(
                f"图片文件过大"
                f"（{size / 1048576:.1f}MB），"
                f"上限 {max_mb}MB。"
            )
        with open(path, "rb") as f:
            return f.read()

    def _validate_dimensions(self, data: bytes) -> None:
        """通过 PIL 头部预检分辨率，防范解压炸弹。

        仅解析图片头部获取尺寸，不触发完整像素解码。
        PIL 无法识别格式时静默跳过（由 _decode 二次校验）。

        Raises:
            ValueError: 尺寸无效或超过像素上限。
        """
        max_pixels = self._config.max_image_pixels
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
            if w <= 0 or h <= 0:
                raise ValueError("图片尺寸无效。")
            if w * h > max_pixels:
                raise ValueError(
                    f"图片分辨率过大"
                    f"（{w}x{h}），"
                    f"上限 {max_pixels} 像素。"
                )
        except (UnidentifiedImageError, OSError):
            # PIL 无法识别或文件损坏 → 由 cv2 解码后检查
            pass
        except ValueError:
            raise
        # MemoryError / SystemError 等不捕获，向上传播

    def _decode(self, data: bytes):
        """从内存字节解码图片，返回 BGR 数组。

        Raises:
            ValueError: 解码失败或分辨率超限。
        """
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                "无法解码图片，"
                "文件可能已损坏或格式不支持。"
            )
        # 二次校验（PIL 预检未覆盖的格式）
        h, w = bgr.shape[:2]
        max_pixels = self._config.max_image_pixels
        if w * h > max_pixels:
            raise ValueError(
                f"图片分辨率过大"
                f"（{w}x{h}），"
                f"上限 {max_pixels} 像素。"
            )
        return bgr

    def process(
        self, model: object, data: bytes, mode: str
    ) -> str:
        """完整处理流程：校验 → 缓存 → 解码 → 推理 → 保存。

        Args:
            model: 已加载的推理模型（调用方负责捕获引用）。
            data: 图片字节快照（由 read_image_bytes 生成）。
            mode: 输出模式（transparent/white/mask/compare）。

        Returns:
            输出文件路径。

        Raises:
            ValueError: 校验或解码失败。
        """
        from anime_cutout_lib import inference as seg

        # 1. 校验分辨率（解压炸弹防护）
        self._validate_dimensions(data)

        # 2. 缓存查询
        cache_enabled = self._config.cache_enabled
        key = _cache_key(data, mode) if cache_enabled else None
        if key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug(f"[cutout] 缓存命中: {key[:12]}")
                return cached

        # 3. 解码（单次，从内存字节）
        bgr = self._decode(data)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 4. 推理
        mask = seg.get_mask(
            model,
            rgb,
            use_amp=self._config.use_amp,
            s=self._config.img_size,
        )
        result = seg.render_result(bgr, mask, mode)

        # 5. 原子保存
        ext = _OUTPUT_EXT.get(mode, ".png")
        out_path = self._save_atomic(result, key, ext)

        # 6. 缓存写入
        if key is not None:
            self._cache.put(key, out_path)
        return out_path

    def _save_atomic(
        self,
        result,
        key: Optional[str],
        ext: str,
    ) -> str:
        """原子保存结果图片：先写临时文件，再重命名。

        os.replace 跨设备失败时回退 shutil.move。
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if key is not None:
            final = self._output_dir / f"{key}{ext}"
        else:
            final = (
                self._output_dir
                / f"out_{uuid.uuid4().hex}{ext}"
            )
        tmp = (
            self._output_dir
            / f".{final.name}.tmp_{uuid.uuid4().hex}"
        )
        ok, buf = cv2.imencode(ext, result)
        if not ok:
            raise ValueError("结果图片编码失败。")
        try:
            buf.tofile(str(tmp))
            try:
                os.replace(str(tmp), str(final))
            except OSError:
                # 跨文件系统回退
                shutil.move(str(tmp), str(final))
        except OSError:
            _safe_remove(str(tmp))
            raise
        return str(final)


# ------------------------------------------------------------------ #
#  MessageExtractor — 从消息事件中提取图片路径
# ------------------------------------------------------------------ #


class MessageExtractor:
    """从消息事件中提取图片本地路径。

    优先级：消息图片 → 回复消息图片 → 文本 URL。
    """

    @staticmethod
    async def find_image(
        event: AstrMessageEvent,
    ) -> Optional[str]:
        """从消息链、回复消息或文本 URL 中提取图片路径。

        Returns:
            图片本地路径，或 None（未找到图片）。
        """
        comps = event.get_messages()

        # 1) 消息中的图片
        for comp in comps:
            if isinstance(comp, Comp.Image):
                try:
                    return await comp.convert_to_file_path()
                except Exception as e:
                    logger.warning(
                        f"[cutout] 图片解析失败: {e}"
                    )

        # 2) 回复消息中的图片
        for comp in comps:
            if isinstance(comp, Comp.Reply):
                chain = getattr(comp, "chain", None)
                if not isinstance(chain, (list, tuple)):
                    continue
                for sub in chain:
                    if isinstance(sub, Comp.Image):
                        try:
                            return (
                                await sub.convert_to_file_path()
                            )
                        except Exception as e:
                            logger.warning(
                                f"[cutout] 回复图片解析失败: {e}"
                            )

        # 3) 消息文本中的图片 URL
        text = str(event.message_str or "")
        for url in re.findall(r"https?://\S+", text):
            if _looks_like_image_url(url):
                try:
                    img = Comp.Image.fromURL(url)
                    return await img.convert_to_file_path()
                except Exception as e:
                    logger.warning(
                        f"[cutout] 图片 URL 下载失败: {e}"
                    )
        return None


# ------------------------------------------------------------------ #
#  AnimeCutoutPlugin — 插件入口，仅负责编排
# ------------------------------------------------------------------ #


class AnimeCutoutPlugin(Star):
    """动漫图片一键抠图插件，基于 ISNet（isnetis.ckpt）。

    本类仅负责组件编排与命令路由，具体逻辑委托给：
    - ModelManager     模型加载与状态
    - CacheManager     LRU 缓存与磁盘清理
    - ImageProcessor   图片校验、解码、推理、保存
    - MessageExtractor 消息图片提取
    - PluginConfig     类型安全的配置访问
    """

    def __init__(
        self, context: Context, config: AstrBotConfig
    ) -> None:
        super().__init__(context)
        self._config = PluginConfig(config)
        self.data_dir = StarTools.get_data_dir(
            "astrbot_plugin_anime_cutout"
        )
        cache_dir = self.data_dir / "cache"
        tmp_dir = self.data_dir / "tmp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._model_mgr = ModelManager(
            self._config, self.data_dir
        )
        self._cache = CacheManager(
            self._config, cache_dir, tmp_dir
        )
        self._processor = ImageProcessor(
            self._config, self._cache, cache_dir
        )
        self._cleanup_stop = asyncio.Event()
        self._cleanup_task: Optional[asyncio.Task] = None

    # --- 依赖检查 -------------------------------------------------

    @staticmethod
    def _check_dependencies() -> None:
        """在 initialize 中提前检查必需依赖，给出友好提示。"""
        deps = [
            ("torch", "torch"),
            ("cv2", "opencv-python-headless"),
            ("PIL", "pillow"),
            ("huggingface_hub", "huggingface_hub"),
        ]
        missing = []
        for mod, pkg in deps:
            try:
                __import__(mod)
            except ImportError:
                missing.append(pkg)
        if missing:
            logger.warning(
                f"[cutout] 缺少依赖: {', '.join(missing)}。"
                f"请运行 pip install {' '.join(missing)}"
            )

    # --- 后台清理任务 ---------------------------------------------

    def _start_cleanup_task(self) -> None:
        """启动后台清理任务（仅 initialize 调用）。"""
        if (
            self._cleanup_task is None
            or self._cleanup_task.done()
        ):
            self._cleanup_stop.clear()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop()
            )

    async def _cleanup_loop(self) -> None:
        """后台定期清理过期文件。

        使用 asyncio.Event 实现优雅退出：
        wait_for 超时 → 执行清理；Event 被 set → 退出循环。
        """
        interval = self._config.cleanup_interval_minutes * 60
        if interval < 60:
            interval = 60
        logger.info(
            f"[cutout] 自动清理任务已启动，"
            f"间隔 {interval // 60} 分钟。"
        )
        while not self._cleanup_stop.is_set():
            try:
                await asyncio.wait_for(
                    self._cleanup_stop.wait(),
                    timeout=interval,
                )
                # Event 被 set，优雅退出
                return
            except asyncio.TimeoutError:
                # 间隔到期，执行清理
                await asyncio.to_thread(self._cache.cleanup_sync)

    # --- 生命周期 -------------------------------------------------

    async def initialize(self) -> None:
        """插件激活时检查依赖并启动后台清理。"""
        self._check_dependencies()
        self._start_cleanup_task()

    async def terminate(self) -> None:
        """插件卸载时优雅停止清理、清理临时文件、释放显存。"""
        # 1. 通知清理循环退出并等待完成
        self._cleanup_stop.set()
        if self._cleanup_task is not None:
            try:
                await asyncio.wait_for(
                    self._cleanup_task, timeout=10
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._cleanup_task = None
        # 2. 最终清理临时文件
        await asyncio.to_thread(self._cache.clear_tmp)
        # 3. 释放模型显存
        await self._model_mgr.release()

    # --- 命令解析 -------------------------------------------------

    def _parse_mode(
        self, mode: str, message_str: str
    ) -> str:
        """从命令参数 / 消息文本解析输出模式。

        未命中则使用配置默认值。
        """
        candidates = []
        if mode:
            candidates.append(str(mode))
        for tok in str(message_str or "").split():
            candidates.append(tok.strip().lstrip("/").lower())
        for c in candidates:
            key = c.lower()
            if key in _MODE_ALIASES:
                return _MODE_ALIASES[key]
        return self._config.output_mode

    # --- 命令 -----------------------------------------------------

    @filter.command("cutout", alias={"抠图", "去背景", "去底"})
    async def cutout(
        self, event: AstrMessageEvent, mode: str = ""
    ):
        '''
        动漫图片一键抠图。发送图片并附带 /cutout 即可去除背景。
        可指定输出模式：/cutout white（白底）、/cutout mask（遮罩）、
        /cutout compare（对比图），默认透明背景。
        支持回复带图片的消息，也支持附带图片链接。
        发送 /cutout reload 可重新加载模型。
        '''
        if str(mode).strip().lower() == "reload":
            self._model_mgr.reset()
            yield event.plain_result(
                "模型已重置，将在下次使用时重新加载。"
            )
            return

        t0 = time.time()
        yield event.plain_result("正在处理…请稍候。")
        mode = self._parse_mode(mode, event.message_str)

        img_path = await MessageExtractor.find_image(event)
        if not img_path:
            yield event.plain_result(
                "未找到图片。请发送「图片 + /cutout」"
                "或回复一条带图片的消息。"
            )
            return

        try:
            # 读取图片到内存（安全快照，防 TOCTOU）
            data = await asyncio.to_thread(
                self._processor.read_image_bytes, img_path
            )
            # 确保模型已加载，捕获引用防 reload 竞态
            model = await self._model_mgr.ensure_model()
            # 处理（传入捕获的 model 引用）
            out_path = await asyncio.to_thread(
                self._processor.process, model, data, mode
            )
        except Exception as e:
            logger.exception("[cutout] 处理失败")
            yield event.plain_result(f"处理失败：{e}")
            return

        yield event.image_result(out_path)
        logger.info(
            f"[cutout] 完成，"
            f"耗时 {time.time() - t0:.1f}s, mode={mode}"
        )

    @filter.command("cutout_help")
    async def cutout_help(self, event: AstrMessageEvent):
        '''查看动漫抠图插件的使用说明与当前配置。'''
        info = {
            "model_path": (
                self._config.model_path or "(自动)"
            ),
            "当前设备": (
                self._model_mgr.model_device or "(未加载)"
            ),
            "img_size": self._config.img_size,
            "默认输出模式": self._config.output_mode,
            "缓存条目": (
                f"{self._cache.cache_size}/"
                f"{self._config.cache_max_entries}"
            ),
            "图片像素上限": self._config.max_image_pixels,
        }
        lines = [
            "动漫抠图 使用说明：",
            "· 发送「图片 + /cutout」→ 透明背景抠图",
            "· /cutout white → 白底",
            "· /cutout mask → 遮罩",
            "· /cutout compare → 对比图",
            "· 可回复带图片的消息触发",
            "· /cutout reload 重新加载模型",
            "",
            "当前配置：",
        ]
        lines.extend(f"{k}: {v}" for k, v in info.items())
        yield event.plain_result("\n".join(lines))

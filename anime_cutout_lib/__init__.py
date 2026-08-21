"""动漫抠图插件核心库。

为了不拖慢 AstrBot 启动，torch / cv2 等重型依赖的导入推迟到首次使用时进行，
因此本包不在此处 import inference。
"""
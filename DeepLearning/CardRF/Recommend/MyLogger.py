import logging
import time

# 配置日志管理
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="/SaveFolders/Recommend/logs/training_log.log",  # 日志保存的文件名
    filemode="a"  # 追加模式，防止覆盖日志
)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台日志级别

# 设置时间为本地时间
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter.converter = time.localtime  # 转换为本地时间
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

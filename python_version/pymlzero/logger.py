import os
from datetime import datetime
import time

class Timer:
    def __init__(self, name):
        self.timer = 0
        self.sum_time = 0
        self.name = name

    def start(self):
        self.timer = time.time() * 1000 

    def end(self):
        end_time = time.time() * 1000
        self.timer = end_time - self.timer
        self.sum_time += self.timer
        return self.timer
    
    def get_sum(self):
        return self.sum_time

    def get_name(self):
        return self.name

class Logger:
    def __init__(self, buffer_size=100, flush_interval=10, log_file="log.log"):
        self.log_file = log_file
        self.buffer_size = buffer_size       # 缓冲区最大条目数
        self.flush_interval = flush_interval # 自动刷新间隔（秒）
        self.buffer = []                     # 日志缓冲区
        self.last_flush_time = time.time()   # 上次刷新时间
        self._init_log_format()

    def _init_log_format(self):
        self.time_format = "%Y-%m-%d %H:%M:%S"
        self.log_template = "{time} [Rank{info_rank}] {message}\n"

    def _flush(self):
        if not self.buffer:
            return
        
        mode = "a" if os.path.exists(self.log_file) else "w"
        with open(self.log_file, mode, encoding="utf-8") as f:
            f.writelines(self.buffer)
        self.buffer.clear()
        self.last_flush_time = time.time()

        
    def log(self, info_rank, info):

        log_entry = self.log_template.format(
            time=datetime.now().strftime(self.time_format),
            info_rank=info_rank,
            message=info
        )
        
        self.buffer.append(log_entry)

        if len(self.buffer) >= self.buffer_size or (time.time() - self.last_flush_time) > self.flush_interval:
            self._flush()

    def close(self):
        self._flush()

    def flush(self):
        self._flush()

    def __del__(self):
        self._flush()

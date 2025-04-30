import os
import re

def delete_specific_weights():
    # 获取当前目录下所有文件
    files = os.listdir()
    
    # 定义匹配weight_id.ckpt文件名的正则表达式
    pattern = re.compile(r'^weight_(\d+)\.ckpt$')
    
    deleted_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            weight_id = int(match.group(1))
            if weight_id % 5 != 0:
                os.remove(file)
                deleted_files.append(file)
    
    print(f"已删除 {len(deleted_files)} 个文件:")
    for file in deleted_files:
        print(f"- {file}")

if __name__ == "__main__":
    delete_specific_weights()
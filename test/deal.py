import re
import csv
import numpy as np


def parse_log_file(file_path):
    # 初始化数据存储
    cpu_times = {}
    gpu_times = {}
    
    # 正则表达式模式，用于匹配CPU和GPU时间
    cpu_pattern = re.compile(r'batchsize: (\d+), cpu time: (\d+\.\d+)')
    gpu_pattern = re.compile(r'batchsize: (\d+), gpu time: (\d+\.\d+)')
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # 提取所有CPU时间和GPU时间
        cpu_matches = cpu_pattern.findall(content)
        gpu_matches = gpu_pattern.findall(content)
        
        # 整理CPU数据
        for match in cpu_matches:
            batch_size = int(match[0])
            time = float(match[1])
            if batch_size not in cpu_times:
                cpu_times[batch_size] = []
            cpu_times[batch_size].append(time)
        
        # 整理GPU数据
        for match in gpu_matches:
            batch_size = int(match[0])
            time = float(match[1])
            if batch_size not in gpu_times:
                gpu_times[batch_size] = []
            gpu_times[batch_size].append(time)
    
    return cpu_times, gpu_times

def calculate_average_times(cpu_times, gpu_times):
    # 计算平均值
    avg_cpu_times = {batch: np.mean(times) for batch, times in cpu_times.items()}
    avg_gpu_times = {batch: np.mean(times) for batch, times in gpu_times.items()}
    
    # 计算标准偏差
    std_cpu_times = {batch: np.std(times) for batch, times in cpu_times.items()}
    std_gpu_times = {batch: np.std(times) for batch, times in gpu_times.items()}
    
    # 计算加速比
    speedup = {}
    for batch in avg_cpu_times:
        if batch in avg_gpu_times:
            speedup[batch] = avg_cpu_times[batch] / avg_gpu_times[batch]
    
    return avg_cpu_times, avg_gpu_times, std_cpu_times, std_gpu_times, speedup

def save_to_csv(batch_sizes, avg_cpu_times, avg_gpu_times, speedup, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_size', 'cpu_time', 'gpu_time', 'speedup'])
        
        for batch in sorted(batch_sizes):
            writer.writerow([batch, avg_cpu_times[batch], avg_gpu_times[batch], speedup[batch]])


def main():
    # 解析文件
    file_path = '/home/work/file/mlzero/test/build/nohup.out'
    cpu_times, gpu_times = parse_log_file(file_path)
    
    # 确保两者有相同的batch sizes
    common_batch_sizes = set(cpu_times.keys()).intersection(set(gpu_times.keys()))
    
    # 计算平均时间和加速比
    avg_cpu_times, avg_gpu_times, std_cpu_times, std_gpu_times, speedup = calculate_average_times(cpu_times, gpu_times)
    
    # 保存为CSV
    output_file = 'performance_data.csv'
    save_to_csv(common_batch_sizes, avg_cpu_times, avg_gpu_times, speedup, output_file)
    
    
    # 打印摘要信息
    print("\n性能数据摘要:")
    print(f"{'批次大小':^10}|{'CPU时间(ms)':^12}|{'GPU时间(ms)':^12}|{'加速比':^8}")
    print("-" * 44)
    for batch in sorted(common_batch_sizes):
        print(f"{batch:^10}|{avg_cpu_times[batch]:^12.2f}|{avg_gpu_times[batch]:^12.2f}|{speedup[batch]:^8.2f}")

if __name__ == "__main__":
    main()
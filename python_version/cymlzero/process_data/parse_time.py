import re
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(file_path):
    train_iterations = []
    train_timestamps = []
    consume_times = []
    loss_sums = []
    start_time = None
    
    # 编译正则表达式模式
    train_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[Rank\d+\] train time train (\d+) consume time: ([\d.]+), loss_sum: ([\d.]+)"
    init_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[Rank\d+\] generate latest.ckpt"
    
    # 打开并读取日志文件
    with open(file_path, 'r') as file:
        for line in file:
            # 首先查找初始时间点
            if start_time is None:
                init_match = re.search(init_pattern, line)
                if init_match:
                    start_time_str = init_match.group(1)
                    start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    print(f"找到初始时间点: {start_time}")
            
            # 然后查找训练迭代行
            train_match = re.search(train_pattern, line)
            if train_match:
                timestamp_str = train_match.group(1)
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                iteration = int(train_match.group(2))
                model_time = float(train_match.group(3))
                loss_sum = float(train_match.group(4))
                
                # 将数据添加到列表中
                train_iterations.append(iteration)
                train_timestamps.append(timestamp)
                consume_times.append(model_time)
                loss_sums.append(loss_sum)
    
    if start_time is None:
        raise ValueError("在日志中未找到'generate latest.ckpt'行，无法确定初始时间点!")
    
    # 计算每次迭代距离初始时间点的实际秒数
    actual_seconds = [(ts - start_time).total_seconds() for ts in train_timestamps]
    
    # 创建 DataFrame
    data = {
        "训练迭代": train_iterations,
        "时间戳": train_timestamps,
        "距初始时间(秒)": actual_seconds,
        "模型报告耗时(秒)": consume_times,
        "损失值": loss_sums
    }
    
    return pd.DataFrame(data)

def calculate_iteration_times(df):
    """计算每次迭代的实际耗时和间隔"""
    df = df.copy()
    
    # 计算每次迭代的实际时间间隔
    df['实际迭代间隔(秒)'] = df['距初始时间(秒)'].diff()
    # 第一个迭代的间隔是它的距离初始时间的秒数
    if not df.empty:
        df.loc[df.index[0], '实际迭代间隔(秒)'] = df.loc[df.index[0], '距初始时间(秒)']
    
    # 计算模型报告的每次迭代耗时差
    df['模型单次迭代耗时(秒)'] = df['模型报告耗时(秒)'].diff()
    # 第一个迭代的模型耗时就是它报告的累计时间
    if not df.empty:
        df.loc[df.index[0], '模型单次迭代耗时(秒)'] = df.loc[df.index[0], '模型报告耗时(秒)']
    
    return df

def main():
    log_file = "../log/log.log"  # 日志文件路径
    
    try:
        # 解析日志文件
        df = parse_log_file(log_file)
        
        # 计算每次迭代的实际耗时和间隔
        df = calculate_iteration_times(df)
        
        # 格式化时间戳列为字符串，便于显示
        df['时间戳'] = df['时间戳'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        
        # 将结果打印为表格
        print("\n训练数据统计表:")
        print(df.to_string(index=False))
        
        # 保存结果到CSV文件
        csv_file = "training_stats_with_time.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n数据已保存到 {csv_file}")
        
    except Exception as e:
        print(f"处理日志时出错: {e}")

if __name__ == "__main__":
    main()
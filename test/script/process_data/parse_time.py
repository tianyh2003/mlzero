import re
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def get_initial_time(main_log_path):
    """获取main.log文件的第一行时间作为初始时间点"""
    try:
        with open(main_log_path, 'r') as file:
            first_line = file.readline().strip()
            # 假设第一行包含格式为'YYYY-MM-DD HH:MM:SS'的时间戳
            time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', first_line)
            if time_match:
                init_time_str = time_match.group(1)
                init_time = datetime.datetime.strptime(init_time_str, '%Y-%m-%d %H:%M:%S')
                print(f"从main.log获取的初始时间点: {init_time}")
                return init_time
            else:
                raise ValueError("无法在main.log的第一行找到时间格式")
    except Exception as e:
        print(f"读取main.log时出错: {e}")
        return None

def parse_learner_file(file_path, init_time=None):
    """解析learner日志文件并提取相关信息，计算与初始时间的差值"""
    # 初始化存储数据的列表
    iterations = []
    losses = []
    times_ms = []
    timestamps = []
    actual_seconds = []  # 相对于初始时间的秒数
    
    # 编译正则表达式模式
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[Rank\d+\] train iter: (\d+), loss: ([\d.]+), time: (\d+) ms"
    
    # 打开并读取日志文件
    with open(file_path, 'r') as file:
        for line in file:
            # 查找匹配的行
            match = re.search(pattern, line)
            if match:
                # 提取匹配的数据
                timestamp_str = match.group(1)
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                iteration = int(match.group(2))
                loss = float(match.group(3))
                time_ms = int(match.group(4))
                
                # 计算与初始时间的差值(秒)
                if init_time is not None:
                    seconds_from_init = (timestamp - init_time).total_seconds()
                else:
                    seconds_from_init = None
                
                # 将数据添加到列表中
                timestamps.append(timestamp_str)  # 保存为字符串格式
                iterations.append(iteration)
                losses.append(loss)
                times_ms.append(time_ms)
                actual_seconds.append(seconds_from_init)
    
    # 创建DataFrame
    data = {
        "时间戳": timestamps,
        "迭代次数": iterations,
        "损失值": losses,
        "模型累计耗时(ms)": times_ms,
        "模型累计耗时(s)": [ms / 1000 for ms in times_ms]  # 将毫秒转换为秒
    }
    
    # 如果有初始时间，添加相对时间列
    if init_time is not None:
        data["距初始时间(s)"] = actual_seconds
    
    df = pd.DataFrame(data)
    
    # 计算模型报告的每次迭代单独消耗时间
    df['模型单次迭代耗时(s)'] = df['模型累计耗时(s)'].diff()
    # 处理第一次迭代
    if len(df) > 0:
        df.loc[df.index[0], '模型单次迭代耗时(s)'] = df.loc[df.index[0], '模型累计耗时(s)']
    
    # 如果有初始时间，计算实际的每次迭代间隔
    if '距初始时间(s)' in df.columns:
        df['实际迭代间隔(s)'] = df['距初始时间(s)'].diff()
        if len(df) > 0:
            df.loc[df.index[0], '实际迭代间隔(s)'] = df.loc[df.index[0], '距初始时间(s)']
    
    return df

def main():
    """主函数：解析文件，生成表格和图表"""
    main_log_file = "../log/main.log"  # 主日志文件路径
    learner_log_file = "../log/learner.log"  # 学习器日志文件路径
    
    # 获取初始时间
    init_time = get_initial_time(main_log_file)
    
    # 解析学习器日志文件
    df = parse_learner_file(learner_log_file, init_time)
    
    # 显示表格
    print("\n训练数据统计表:")
    columns_to_show = ["迭代次数", "损失值", "模型累计耗时(s)", "模型单次迭代耗时(s)"]
    if '距初始时间(s)' in df.columns:
        columns_to_show.extend(["距初始时间(s)", "实际迭代间隔(s)"])
    print(df[columns_to_show].to_string(index=False))
    
    # 保存到CSV文件
    csv_file = "learner_stats_with_time.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n数据已保存到 {csv_file}")
    
    # 绘制图表
    plt.figure(figsize=(12, 15))
    sns.set_style("whitegrid")
    
    # 1. 损失值随迭代次数变化
    plt.subplot(3, 1, 1)
    sns.lineplot(x='迭代次数', y='损失值', data=df, marker='o')
    plt.title('训练损失值随迭代次数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True)
    
    # 2. 模型报告的单次迭代耗时变化
    plt.subplot(3, 1, 2)
    sns.lineplot(x='迭代次数', y='模型单次迭代耗时(s)', data=df, marker='o', color='green')
    plt.title('模型报告的单次迭代耗时随迭代次数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('耗时(秒)')
    plt.grid(True)
    
    # 3. 如果有初始时间，添加第三个图表显示实际迭代间隔
    if '实际迭代间隔(s)' in df.columns:
        plt.subplot(3, 1, 3)
        sns.lineplot(x='迭代次数', y='实际迭代间隔(s)', data=df, marker='o', color='orange')
        plt.title('实际迭代间隔随迭代次数变化')
        plt.xlabel('迭代次数')
        plt.ylabel('间隔时间(秒)')
        plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('learner_metrics_with_time.png')
    print("图表已保存到 learner_metrics_with_time.png")
    
    # 尝试显示图形（如果在有图形界面的环境中）
    try:
        plt.show()
    except:
        pass

    # 计算一些统计信息
    avg_model_time = df['模型单次迭代耗时(s)'].mean()
    avg_loss = df['损失值'].mean()
    min_loss = df['损失值'].min()
    min_loss_iter = df.loc[df['损失值'].idxmin(), '迭代次数']
    
    print(f"\n统计摘要:")
    print(f"总训练迭代次数: {len(df)}")
    print(f"平均模型单次迭代耗时: {avg_model_time:.2f}秒")
    
    if '实际迭代间隔(s)' in df.columns:
        avg_actual_interval = df['实际迭代间隔(s)'].mean()
        print(f"平均实际迭代间隔: {avg_actual_interval:.2f}秒")
    
    print(f"平均损失值: {avg_loss:.6f}")
    print(f"最小损失值: {min_loss:.6f} (迭代 {min_loss_iter})")

if __name__ == "__main__":
    main()
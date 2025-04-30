import csv
from datetime import datetime

# 打开源文件和目标文件
with open('eval.log', 'r') as infile, open('1.csv', 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(['耗时（分钟）', 'elo'])  # 写入表头

    # 读取第一行以获取起点时间
    first_line = infile.readline()
    start_time = datetime.strptime(first_line.split(' ')[0] + ' ' + first_line.split(' ')[1], "%Y-%m-%d %H:%M:%S")

    # 重置文件指针，重新逐行读取
    infile.seek(0)

    for line in infile:
        # 提取日期和时间部分
        date_str = line.split(' ')[0]
        time_str = line.split(' ')[1]
        current_time = datetime.strptime(date_str + ' ' + time_str, "%Y-%m-%d %H:%M:%S")
        
        # 计算耗时（以分钟为单位）
        time_delta = current_time - start_time
        elapsed_minutes = time_delta.total_seconds() / 60
        
        # 提取elo分
        if 'elo:' in line:
            elo = float(line.split('elo: ')[1].strip())
            # 写入耗时和elo到CSV文件
            csv_writer.writerow([elapsed_minutes, elo])

print("数据已成功写入1.csv")
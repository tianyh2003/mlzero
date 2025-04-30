import os
import re
import csv

def extract_data_from_file(filepath, pattern):
    with open(filepath, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    return None

def process_log_folder(folder_path):
    data = {}
    
    # # Extract thread_num from actor_0.log
    # actor_log_path = os.path.join(folder_path, 'actor_0.log')
    # if os.path.exists(actor_log_path):
    #     thread_num = extract_data_from_file(actor_log_path, r'thread is (\d+)')
    #     data['thread_num'] = thread_num
    
    # Extract all_time from main.log
    main_log_path = os.path.join(folder_path, 'main.log')
    if os.path.exists(main_log_path):
        all_time = extract_data_from_file(main_log_path, r'program time: (\d+) ms')
        data['all_time'] = all_time
    
    # Extract predict_time from predict_network.log
    predict_log_path = os.path.join(folder_path, 'predict_network.log')
    if os.path.exists(predict_log_path):
        predict_time = extract_data_from_file(predict_log_path, r'predict buffer time: (\d+) ms')
        data['predict_time'] = predict_time
    
    # Extract train_time from learner log
    learner_log_path = os.path.join(folder_path, 'learner')
    if os.path.exists(learner_log_path):
        train_time = extract_data_from_file(learner_log_path, r'train iter: 3, loss: [\d.]+, time: (\d+) ms')
        data['train_time'] = train_time
    
    return data

def main():
    base_path = '/home/work/file/mlzero/logall'
    all_data = []
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            folder_data = process_log_folder(folder_path)
            folder_data['folder'] = folder_name
            all_data.append(folder_data)
    
    # Write data to CSV file
    csv_file = '/home/work/file/mlzero/logall_summary.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['folder', 'thread_num', 'all_time', 'predict_time', 'train_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in all_data:
            writer.writerow(data)
    
    print(f"Data has been written to {csv_file}")

if __name__ == "__main__":
    main()
import csv
import re
# 定义两个CSV文件路径
csv1_path = 'gold_codet5.csv'
csv2_path = 'pred_codet5.csv'

# 读取第一个CSV文件的第一列数据
csv1_data = set()
with open(csv1_path, 'r', newline='') as csvfile1:
    reader1 = csv.reader(csvfile1)
    for row in reader1:
        if row:
            csv1_data.add(row[0])

# 读取第二个CSV文件并输出第一列与第一个CSV文件相同的行及行号
with open(csv2_path, 'r', newline='') as csvfile2:
    reader2 = csv.reader(csvfile2)
    line_number = 0
    for row in reader2:
        line_number += 1
        if row[0] in csv1_data:
            words = re.findall(r'\w+', row[0])  # 使用正则表达式提取单词
            if len(words) > 4:
                print(f"Line {line_number}: {row}")
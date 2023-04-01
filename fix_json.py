import os
import re
import json

def escape_backslashes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 将所有单独的反斜杠替换为双反斜杠
    escaped_content = content.replace('\\', '\\\\')

    # 将修复后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(escaped_content)

    print(f"成功将反斜杠转义为双反斜杠: {file_path}")

# 使用以下代码调用 escape_backslashes 函数，传入您的 JSON 文件路径

def process_all_json_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                escape_backslashes(file_path)

# 调用 process_all_json_files_in_directory 函数，传入包含 JSON 文件的文件夹路径
'''process_all_json_files_in_directory(r'E:\个人研究\learn_dl-master\log\2023-03-20-21-35-50')'''


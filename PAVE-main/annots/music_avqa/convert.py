import json

# JSON文件路径
json_file_path = "/mnt/sda/shenhao/code/AVQA/PAVE-main/annots/music_avqa/music_avqa_all_videos_mapping.json"

# 读取JSON文件
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON解码错误: {file_path}")

# 修改JSON内容
def modify_json(data):
    prefix = "/mnt/sda/shenhao/datasets/MUSIC-AVQA"
    modified_data = {}
    for key, value in data.items():
        if isinstance(value, str) and not value.startswith(prefix):
            if value.startswith("MUSIC-AVQA-videos-Real"):
                value = value.replace("MUSIC-AVQA-videos-Real", prefix)
            elif value.startswith("MUCIS-AVQA-videos-Synthetic"):
                value = value.replace("MUCIS-AVQA-videos-Synthetic", prefix)
        modified_data[key] = value
    return modified_data

if __name__ == "__main__":
    json_data = read_json(json_file_path)
    if json_data:
        new_json_data = modify_json(json_data)
        print("修改后的JSON数据：")
        print(new_json_data)
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(new_json_data, f, indent=4, ensure_ascii=False)
            print("保存成功！")
        except Exception as e:
            print(f"保存失败: {e}")
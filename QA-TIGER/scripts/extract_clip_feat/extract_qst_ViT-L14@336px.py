import os
import torch
import numpy as np
import json

# 导入项目中的 CLIP 模型模块

# 设置设备：默认尝试使用 cuda:2（适用于多卡环境），若不可用则使用 CPU
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# 加载预训练的 ViT-L/14@336px 模型及其图像预处理方法
model, preprocess = scripts.extract_clip_feat.clip_net.clip.load("ViT-L/14@336px", device=device)


def qst_feat_extract(qst):
    """
    使用 CLIP 模型对问题文本进行编码，提取语义特征向量

    参数:
        qst (str): 问题文本字符串

    返回:
        text_features (torch.Tensor): 提取的文本特征向量
    """
    # 对文本进行 token 化并移动到 GPU
    text = scripts.extract_clip_feat.clip_net.clip.tokenize(qst).to(device)

    # 不计算梯度以提升推理速度
    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features


def QstCLIP_feat(json_path, dst_qst_path):
    """
    从 JSON 文件中读取问题，并为每个问题生成 CLIP 特征向量并保存为 .npy 文件

    参数:
        json_path (str): 输入 JSON 文件路径
        dst_qst_path (str): 输出特征文件保存路径
    """
    # 读取 JSON 数据
    samples = json.load(open(json_path, 'r'))

    # 初始化词表，用于记录所有出现的问题词汇
    ques_vocab = ['<pad>']

    i = 0
    for sample in samples:
        i += 1
        # 获取问题内容并分词
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]  # 去掉最后标点符号

        # 获取问题 ID 并构建保存路径
        question_id = sample['question_id']
        print("\n")
        print("question id: ", question_id)

        save_file = os.path.join(dst_qst_path, str(question_id) + '.npy')

        # 如果该问题特征已存在，则跳过
        if os.path.exists(save_file):
            print(question_id, " is already exist!")
            continue

        p = 0
        # 替换模板变量（如 <object>）为实际值
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = sample['templ_values'][p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        # 合并问题词并添加问号
        question = ' '.join(question) + '?'
        print(question)

        # 提取问题特征
        qst_feat = qst_feat_extract(question)
        print(qst_feat.shape)

        # 转换为 NumPy 格式并保存
        qst_features = qst_feat.float().cpu().numpy()
        np.save(save_file, qst_features)


if __name__ == "__main__":
    """
    主程序入口：
    - json_path: 包含问题数据的 JSON 文件路径
    - dst_qst_path: 存储提取出的问题特征的目录路径
    """
    # json_path = "/data/AVQA/data/sub-qst-test.json"
    json_path = "/mnt/sda/shenhao/code/MUSIC-AVQA/data/json_update/avqa-train.json"

    # dst_qst_path = "/data/AVQA/LLM-AVQA/sub-qst-feat"
    dst_qst_path = "/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/sub-qst-feat"


    QstCLIP_feat(json_path, dst_qst_path)


    
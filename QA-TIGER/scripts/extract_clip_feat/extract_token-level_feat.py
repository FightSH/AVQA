# 导入必要的库
import os
import torch
from PIL import Image
import numpy as np
import glob

# 导入CLIP模型相关模块

# 设置设备（优先使用GPU，否则使用CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = scripts.extract_clip_feat.clip_net.clip.load("ViT-B/32", device=device)

def clip_feat_extract(img):
    """
    使用CLIP模型提取图像特征

    参数:
        img (str): 图像文件路径

    返回:
        torch.Tensor: 提取的图像特征向量
    """
    # 使用预处理函数加载图像并转换为张量
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        # 提取图像特征
        image_features = model.encode_image(image)
    return image_features

def ImageClIP_Patch_feat_extract(dir_fps_path, dst_clip_path):
    """
    从视频帧中提取CLIP图像特征，并保存为npy文件

    参数:
        dir_fps_path (str): 包含视频帧的目录路径
        dst_clip_path (str): 特征保存的目标目录路径
    """
    # 获取视频列表
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)

        # 构造保存文件路径
        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        # 获取视频帧列表
        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        
        # 计算采样帧数
        params_frames = len(video_img_list)
        samples = np.round(np.linspace(0, params_frames-1, params_frames))

        # 提取采样帧
        img_list = [video_img_list[int(sample)] for sample in samples]
        img_features = torch.zeros(len(img_list), patch_nums, C)

        idx = 0
        for img_cont in img_list:
            # 提取单帧特征
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        # 将特征转换为numpy数组并保存
        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ", img_features.shape)

def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):
    """
    从视频帧中提取CLIP图像特征，并保存为npy文件

    参数:
        dir_fps_path (str): 包含视频帧的目录路径
        dst_clip_path (str): 特征保存的目标目录路径
    """
    # 获取视频列表
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx += 1
        print("\n--> ", video_idx, video)

        # 构造保存文件路径
        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        # 获取视频帧列表
        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        
        # 计算采样帧数
        params_frames = len(video_img_list)
        samples = np.round(np.linspace(0, params_frames-1, params_frames))

        # 提取采样帧
        img_list = [video_img_list[int(sample)] for sample in samples]
        img_features = torch.zeros(len(img_list), C)

        idx = 0
        for img_cont in img_list:
            # 提取单帧特征
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        # 将特征转换为numpy数组并保存
        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ", img_features.shape)

def qst_feat_extract(qst):
    """
    使用CLIP模型提取问题文本特征

    参数:
        qst (str): 问题文本

    返回:
        torch.Tensor: 提取的问题特征向量
    """
    # 对问题文本进行分词并转换为张量
    text = clip.tokenize(qst).to(device)
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        # 提取文本特征
        text_features = model.encode_text(text)
    return image_features

def QstCLIP_feat(json_path, dst_qst_path):
    """
    从JSON文件中提取问题特征并保存

    参数:
        json_path (str): 包含问题的JSON文件路径
        dst_qst_path (str): 特征保存的目标目录路径
    """
    # 加载JSON文件
    samples = json.load(open(json_path, 'r'))
    
    ques_vocab = ['<pad>']
    # ans_vocab = []

    i = 0
    for sample in samples:
        i += 1
        # 处理问题文本
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        print(question)

if __name__ == "__main__":
    # 定义全局变量
    patch_nums = 50
    C = 512

    # 设置输入和输出路径
    dir_fps_path = '/data/MUSIC-AVQA/avqa-frames-1fps'
    dst_clip_path = '/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/clip_vit_b32'
    # 调用函数提取特征
    ImageClIP_feat_extract(dir_fps_path, dst_clip_path)
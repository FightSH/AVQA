import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import ClapAudioModelWithProjection, AutoProcessor
import warnings

# 忽略一些不影响结果的警告
warnings.filterwarnings("ignore")

# --- 1. 参数设置 ---
# 音频参数
AUDIO_DURATION_SECONDS = 60
TARGET_SAMPLE_RATE = 48000  # HTS-AT 模型通常使用的采样率

# 滑动窗口参数
WINDOW_SECONDS = 1  # 窗口大小（秒），保留足够上下文
STRIDE_SECONDS = 1   # 步长（秒），我们希望每秒得到一个特征

# --- 2. 准备工作：加载模型并创建一个虚拟音频文件 ---

def setup():
    """加载模型和处理器，并创建一个用于演示的音频文件。"""
    print("正在加载预训练的 HTS-AT 模型和特征提取器...")
    
    # 从Hugging Face Hub加载预训练模型和对应的特征提取器（处理器）
    model_id = "laion/clap-htsat-fused"
    processor = AutoProcessor.from_pretrained(model_id,cache_dir="/mnt/sda/shenhao/models")
    model = model = ClapAudioModelWithProjection.from_pretrained(model_id,cache_dir="/mnt/sda/shenhao/models")
    
    # 将模型设置为评估模式
    model.eval()
    print("模型加载完毕。")

    # 创建一个60秒的虚拟音频文件用于演示
    print(f"正在创建一个 {AUDIO_DURATION_SECONDS} 秒的虚拟音频文件...")
    dummy_audio_data = np.random.randn(AUDIO_DURATION_SECONDS * TARGET_SAMPLE_RATE)
    dummy_audio_path = "dummy_audio_60s.wav"
    sf.write(dummy_audio_path, dummy_audio_data, TARGET_SAMPLE_RATE)
    print(f"虚拟音频文件已保存至: {dummy_audio_path}")
    
    return model, processor, dummy_audio_path

def extract_features_per_second(audio_path, model, processor):
    """
    使用滑动窗口从音频文件中提取每秒的特征。
    """
    print("\n开始提取特征...")
    
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if sr != processor.feature_extractor.sampling_rate:
            print(f"音频原始采样率为 {sr}Hz，正在重采样至 {processor.feature_extractor.sampling_rate}Hz...")
            y = librosa.resample(y, orig_sr=sr, target_sr=processor.feature_extractor.sampling_rate)
    except Exception as e:
        print(f"加载音频文件时出错: {e}")
        return None

    window_samples = WINDOW_SECONDS * TARGET_SAMPLE_RATE
    stride_samples = STRIDE_SECONDS * TARGET_SAMPLE_RATE
    
    all_second_embeddings = []

    num_windows = (len(y) - window_samples) // stride_samples + 1
    for i in range(num_windows):
        start_sample = i * stride_samples
        end_sample = start_sample + window_samples
        window_audio = y[start_sample:end_sample]
        
        print(f"处理窗口 {i+1}/{num_windows} (时间: {i}-{i+WINDOW_SECONDS}s)...")

        # 使用处理器将音频转换为模型输入
        inputs = processor(
            audios=window_audio, 
            sampling_rate=TARGET_SAMPLE_RATE, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            # 获取音频特征嵌入
            # CLAP模型直接输出最终的特征，我们不需要访问隐藏层
            # 注意：这里的输出维度是 512，这是CLAP模型经过投影后的维度
            audio_features = model(**inputs).audio_embeds

        # 对于CLAP模型，它通常会为一段音频输出一个池化后的特征
        # 如果需要时间序列特征，我们需要修改模型或者使用不同的模型
        # 但为了实现每秒一个特征，我们可以直接将当前窗口的特征视为这一秒的代表
        # 这是一个简化，但对于很多任务是有效的
        
        # 将当前窗口的池化后特征作为这一秒的特征
        # CLAP模型的 `.audio_embeds` 已经是池化后的结果, 形状为 (1, 512)
        # 注意：这里的维度是512，而不是768，因为CLAP在HTS-AT之上加了一个投影层
        second_embedding = audio_features[0]
        
        all_second_embeddings.append(second_embedding.numpy())

    final_embeddings = np.array(all_second_embeddings)
    final_embeddings = final_embeddings[:AUDIO_DURATION_SECONDS]
    
    return final_embeddings

# --- 3. 主执行流程 ---
if __name__ == "__main__":
    # 执行准备工作
    model, feature_extractor, audio_path = setup()
    
    # 执行特征提取
    final_feature_matrix = extract_features_per_second(audio_path, model, feature_extractor)
    
    # 验证结果
    if final_feature_matrix is not None:
        print("\n--- 特征提取完成 ---")
        print(f"最终得到的特征矩阵形状: {final_feature_matrix.shape}")
        
        if final_feature_matrix.shape == (60, 512):
            print("成功！输出形状符合预期 (60, 512)。")
        else:
            print(f"警告：输出形状与预期不符。请检查音频时长或代码逻辑。")
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # set gpu number
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import ClapAudioModelWithProjection, AutoProcessor
import warnings
import contextlib
import wave

# 忽略一些不影响结果的警告
warnings.filterwarnings("ignore")

# --- 1. 参数设置 ---
# 音频参数
AUDIO_DURATION_SECONDS = 60
TARGET_SAMPLE_RATE = 48000  # HTS-AT 模型通常使用的采样率

# 滑动窗口参数
WINDOW_SECONDS = 1  # 窗口大小（秒），保留足够上下文
STRIDE_SECONDS = 1   # 步长（秒），我们希望每秒得到一个特征

# 文件路径设置
audio_dir = "/mnt/sda/shenhao/datasets/MUSIC-AVQA/audio/"  # .wav audio files
save_dir = "/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/audit_feat/60hts_at"

# get audio length
def get_audio_len(audio_file):
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = int(frames / float(rate))
        return wav_length

# 添加处理短音频的函数
def process_audio_with_padding(audio_file, target_length=60):
    """处理音频文件，如果长度不足则用最后1秒填充"""
    y, sr = librosa.load(audio_file, sr=TARGET_SAMPLE_RATE)
    
    # 如果音频长度不足target_length秒
    if len(y) < sr * target_length:
        # 获取实际音频长度（秒）
        actual_length = len(y) / sr
        # 需要填充的秒数
        padding_needed = target_length - actual_length
        
        # 获取最后1秒的音频数据用于填充
        if len(y) > sr:  # 确保音频至少有1秒
            last_second = y[-sr:]
        else:  # 音频不足1秒，就用全部内容
            last_second = y
        
        # 计算需要重复的次数
        repeat_times = int(np.ceil(padding_needed))
        
        # 创建填充数据
        padding_data = np.tile(last_second, repeat_times)
        needed_samples = int(padding_needed * sr)
        padding_data = padding_data[:needed_samples]
        
        # 合并原始音频和填充数据
        padded_audio = np.concatenate((y, padding_data))
        
        return padded_audio
    else:
        # 音频长度足够，截取前60秒
        return y[:sr * target_length]

def setup():
    """加载模型和处理器"""
    print("正在加载预训练的 HTS-AT 模型和特征提取器...")
    
    # 从Hugging Face Hub加载预训练模型和对应的特征提取器（处理器）
    model_id = "laion/clap-htsat-fused"
    processor = AutoProcessor.from_pretrained(model_id, cache_dir="/mnt/sda/shenhao/models")
    model = ClapAudioModelWithProjection.from_pretrained(model_id, cache_dir="/mnt/sda/shenhao/models")
    
    # 将模型移动到GPU并设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"模型加载完毕，使用设备: {device}")
    
    return model, processor

def extract_features_per_second(audio_path, model, processor):
    """
    使用滑动窗口从音频文件中提取每秒的特征。
    """
    try:
        # 获取音频长度
        num_secs_real = get_audio_len(audio_path)
        
        # 处理音频（填充或截取）
        if num_secs_real < AUDIO_DURATION_SECONDS:
            print(f"音频长度不足{AUDIO_DURATION_SECONDS}秒，使用最后1秒填充至{AUDIO_DURATION_SECONDS}秒")
            y = process_audio_with_padding(audio_path, AUDIO_DURATION_SECONDS)
        else:
            y, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
            y = y[:TARGET_SAMPLE_RATE * AUDIO_DURATION_SECONDS]  # 截取前60秒
            
    except Exception as e:
        print(f"加载音频文件时出错: {e}")
        return None

    window_samples = WINDOW_SECONDS * TARGET_SAMPLE_RATE
    stride_samples = STRIDE_SECONDS * TARGET_SAMPLE_RATE
    
    all_second_embeddings = []

    num_windows = AUDIO_DURATION_SECONDS  # 固定为60个窗口
    for i in range(num_windows):
        start_sample = i * stride_samples
        end_sample = start_sample + window_samples
        window_audio = y[start_sample:end_sample]
        
        # 使用处理器将音频转换为模型输入
        inputs = processor(
            audios=window_audio, 
            sampling_rate=TARGET_SAMPLE_RATE, 
            return_tensors="pt"
        )
        
        # 将输入数据移动到GPU
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 获取音频特征嵌入
            audio_features = model(**inputs).audio_embeds

        # 将当前窗口的池化后特征作为这一秒的特征
        second_embedding = audio_features[0]
        all_second_embeddings.append(second_embedding.cpu().numpy())

    final_embeddings = np.array(all_second_embeddings)
    
    return final_embeddings

# --- 3. 主执行流程 ---
if __name__ == "__main__":
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载模型
    model, processor = setup()
    
    # 获取音频文件列表
    lis = sorted(os.listdir(audio_dir))
    len_data = len(lis)
    print(f"总共需要处理 {len_data} 个音频文件")

    i = 0
    
    for n in range(len_data):
        i += 1
        
        # 检查文件是否为wav格式
        if not lis[n].endswith('.wav'):
            continue
            
        # save file
        outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
        if os.path.exists(outfile):
            print(f"\nProcessing: {i} / {len_data} ----> {lis[n][:-4]}.npy is already exist!")
            continue

        # 音频文件路径
        audio_index = os.path.join(audio_dir, lis[n])
        num_secs_real = get_audio_len(audio_index)
        print(f"\nProcessing: {i} / {len_data} --------> video: {lis[n]} ---> sec: {num_secs_real}")
        
        # 执行特征提取
        final_feature_matrix = extract_features_per_second(audio_index, model, processor)
        
        # 验证并保存结果
        if final_feature_matrix is not None:
            np.save(outfile, final_feature_matrix)
            print(f" save info: {lis[n][:-4]}.npy ---> {final_feature_matrix.shape}")
            
            if final_feature_matrix.shape[0] == 60:
                print("成功！输出形状符合预期。")
            else:
                print(f"警告：输出形状与预期不符。实际形状: {final_feature_matrix.shape}")
        else:
            print(f"处理文件 {lis[n]} 时出错")

    print("\n---------------------------------- end ----------------------------------\n")
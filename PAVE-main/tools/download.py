import os
import requests

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfFolder

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_specific_file(repo_id, filename, save_path):
    """下载数据集仓库中的特定文件"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=save_path,
            revision='main'
        )
        print(f"文件已下载到: {file_path}")
        return file_path
    except Exception as e:
        print(f"下载文件时出错: {e}")
        return None


def download_file_from_hf(url, save_path):
    # 将 blob 替换为 resolve
    download_url = url.replace('/blob/', '/resolve/')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    resp = requests.get(download_url, stream=True)
    resp.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"文件已下载到: {save_path}")

def check_model_exists(model_id: str, token: str = None):
    """检查模型是否存在"""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, token=token)
        print(f"✅ 模型存在: {model_id}")
        print(f"📝 模型信息: {info.modelId}")
        return True
    except Exception as e:
        print(f"❌ 模型检查失败: {str(e)}")
        return False

def test_connection():
    """测试网络连接"""
    import requests
    try:
        response = requests.get("https://hf-mirror.com", timeout=10)
        if response.status_code == 200:
            print("✅ 网络连接正常")
            return True
        else:
            print(f"❌ 网络连接异常，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 网络连接测试失败: {str(e)}")
        return False

def clear_cache(model_id: str = None):
    """清理Hugging Face缓存"""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        
        if model_id:
            # 删除特定模型的缓存
            for repo in cache_info.repos:
                if model_id in repo.repo_id:
                    print(f"🗑️ 清理模型缓存: {repo.repo_id}")
                    repo.delete()
        else:
            print(f"📁 缓存目录: {cache_info.size_on_disk_str}")
            print("提示: 可以手动删除缓存目录来清理所有缓存")
            
    except Exception as e:
        print(f"⚠️ 缓存清理失败: {str(e)}")

def download_model_weights(
    model_id: str,
    local_dir: str = None,
    token: str = None,
    allow_patterns: list = None,
    ignore_patterns: list = None,
    force_download: bool = False
):
    """
    从Hugging Face Hub下载模型权重
    
    参数:
        model_id (str): 模型ID，格式为 "username/model-name"
        local_dir (str): 本地保存目录，默认为 "./models/{model_name}"
        token (str): Hugging Face访问令牌（如果需要）
        allow_patterns (list): 允许下载的文件模式列表
        ignore_patterns (list): 忽略下载的文件模式列表
        force_download (bool): 是否强制重新下载
    """
    
    print("🔍 开始诊断...")
    
    # 1. 测试网络连接
    if not test_connection():
        print("请检查网络连接后重试")
        return None
    
    # 2. 检查模型是否存在
    if not check_model_exists(model_id, token):
        print("请检查模型ID是否正确，或者是否需要登录token")
        return None

    # 设置默认保存目录
    if local_dir is None:
        model_name = model_id.split('/')[-1]
        local_dir = f"./models/{model_name}"
    
    # 创建目录
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载模型: {model_id}")
    print(f"保存路径: {os.path.abspath(local_dir)}")
    
    try:
        # 下载整个模型仓库
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download,
            resume_download=True  # 支持断点续传
        )
        
        print(f"✅ 模型下载完成！")
        print(f"📁 文件保存在: {downloaded_path}")
        
        # 显示下载的文件列表
        print("\n📋 下载的文件:")
        for root, dirs, files in os.walk(downloaded_path):
            level = root.replace(downloaded_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                print(f"{subindent}{file} ({size_mb:.2f} MB)")
        
        return downloaded_path
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return None

def download_specific_files(model_id: str, filenames: list, local_dir: str = None, token: str = None):
    """
    下载模型的特定文件
    
    参数:
        model_id (str): 模型ID
        filenames (list): 要下载的文件名列表
        local_dir (str): 本地保存目录
        token (str): Hugging Face访问令牌
    """
    
    if local_dir is None:
        model_name = model_id.split('/')[-1]
        local_dir = f"./models/{model_name}"
    
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载特定文件: {model_id}")
    print(f"文件列表: {filenames}")
    
    downloaded_files = []
    
    for filename in filenames:
        try:
            print(f"正在下载: {filename}")
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=local_dir,
                token=token,
                resume_download=True
            )
            downloaded_files.append(file_path)
            print(f"✅ {filename} 下载完成")
            
        except Exception as e:
            print(f"❌ {filename} 下载失败: {str(e)}")
    
    return downloaded_files

# 示例调用



def download():
    """主函数"""
    
    # 目标模型配置
    # MODEL_ID = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    MODEL_ID = "lmms-lab/llava-onevision-qwen2-7b-ov"
    
    print("🤗 Hugging Face模型下载工具")
    print("=" * 50)
    
    # 选择下载模式
    # print("请选择下载模式:")
    # print("1. 下载完整模型 (推荐)")
    # print("2. 仅下载权重文件")
    # print("3. 自定义文件下载")
    
    # choice = input("请输入选择 (1-3): ").strip()
    choice = "1"  # 默认选择下载完整模型
    
    # # 获取访问令牌（可选）
    # token = input("请输入Hugging Face Token (可选，直接回车跳过): ").strip()
    # if not token:
    #     token = None
    token = None
    # 获取保存路径
    # save_dir = input(f"请输入保存路径 (默认: ./models/llava-onevision-qwen2-0.5b-ov): ").strip()
    # save_dir = '/mnt/sda/shenhao/models/llava-onevision-qwen2-0.5b-ov'  # 默认保存路径
    save_dir = '/mnt/sda/shenhao/models/llava-onevision-qwen2-7b-ov'  # 默认保存路径
    if not save_dir:
        save_dir = None
    
    if choice == "1":
        # 下载完整模型
        download_model_weights(
            model_id=MODEL_ID,
            local_dir=save_dir,
            token=token
        )
        
    elif choice == "2":
        # 仅下载权重文件
        download_model_weights(
            model_id=MODEL_ID,
            local_dir=save_dir,
            token=token,
            allow_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth", "config.json", "tokenizer*"]
        )
        
    elif choice == "3":
        # 自定义文件下载
        print("常见文件类型:")
        print("- *.safetensors (推荐的权重格式)")
        print("- *.bin (PyTorch权重文件)")
        print("- config.json (模型配置)")
        print("- tokenizer.json, tokenizer_config.json (分词器)")
        
        files_input = input("请输入要下载的文件名，用逗号分隔: ").strip()
        if files_input:
            filenames = [f.strip() for f in files_input.split(',')]
            download_specific_files(
                model_id=MODEL_ID,
                filenames=filenames,
                local_dir=save_dir,
                token=token
            )
        else:
            print("❌ 未指定文件名")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    download()


# if __name__ == "__main__":
#     url = "https://hf-mirror.com/datasets/zhuomingliu/PAVEDataset/blob/main/MUSIC-AVQA-videos-Real_audio_imagebind_feat.zip"
#     save_path = "/mnt/sda/shenhao/datasets/PAVE/MUSIC-AVQA-videos-Real_audio_imagebind_feat.zip"
#     download_file_from_hf(url, save_path)

# if __name__ == "__main__":
#     # 下载特定的ZIP文件
#     download_specific_file(
#         repo_id="zhuomingliu/PAVEDataset",
#         filename="MUSIC-AVQA-videos-Real_audio_imagebind_feat.zip",
#         save_path="/mnt/sda/shenhao/datasets/PAVE/"
#     )
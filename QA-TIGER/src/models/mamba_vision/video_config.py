#!/usr/bin/env python3

"""
Video MambaVision 配置文件
包含不同任务和数据集的配置
"""

# 基础配置
BASE_CONFIG = {
    'model': {
        'hidden_dim': 256,
        'depths': [2, 2, 8, 2],
        'num_heads': [4, 8, 16, 32],
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'layer_scale': 1e-6,
        'causal': False,
    },
    'training': {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 100,
        'warmup_epochs': 10,
        'scheduler': 'cosine',
    },
    'data': {
        'seq_len': 30,  # 30帧，约1秒@30fps
        'img_dim': 2048,  # ResNet-50 特征
        'audio_dim': 128,  # 音频特征
    }
}

# Kinetics-400 动作识别配置
KINETICS400_CONFIG = {
    **BASE_CONFIG,
    'model': {
        **BASE_CONFIG['model'],
        'num_classes': 400,
        'causal': False,  # 可以看到未来帧
    },
    'data': {
        **BASE_CONFIG['data'],
        'seq_len': 32,  # 稍长的序列
        'img_dim': 2048,  # ResNet-50
        'audio_dim': 128,
    },
    'training': {
        **BASE_CONFIG['training'],
        'batch_size': 12,  # 较大的序列长度，减小batch size
        'learning_rate': 5e-5,
        'epochs': 150,
    }
}

# UCF-101 动作识别配置
UCF101_CONFIG = {
    **BASE_CONFIG,
    'model': {
        **BASE_CONFIG['model'],
        'num_classes': 101,
        'hidden_dim': 192,  # 较小的模型
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
    },
    'data': {
        **BASE_CONFIG['data'],
        'seq_len': 16,  # 较短的序列
    },
    'training': {
        **BASE_CONFIG['training'],
        'batch_size': 24,
        'learning_rate': 1e-4,
        'epochs': 80,
    }
}

# 实时视频理解配置（因果模型）
REALTIME_CONFIG = {
    **BASE_CONFIG,
    'model': {
        **BASE_CONFIG['model'],
        'causal': True,  # 因果模型，不能看到未来
        'hidden_dim': 128,  # 更小的模型以提高速度
        'depths': [1, 2, 4, 2],
        'num_heads': [2, 4, 8, 16],
    },
    'data': {
        **BASE_CONFIG['data'],
        'seq_len': 8,  # 更短的序列以降低延迟
    },
    'training': {
        **BASE_CONFIG['training'],
        'batch_size': 32,
        'learning_rate': 2e-4,
    }
}

# 长视频理解配置
LONG_VIDEO_CONFIG = {
    **BASE_CONFIG,
    'model': {
        **BASE_CONFIG['model'],
        'hidden_dim': 384,  # 更大的模型处理复杂长视频
        'depths': [3, 3, 12, 3],
        'num_heads': [6, 12, 24, 48],
    },
    'data': {
        **BASE_CONFIG['data'],
        'seq_len': 64,  # 长序列
        'img_dim': 1024,  # 可能使用更小的图像特征以节省内存
    },
    'training': {
        **BASE_CONFIG['training'],
        'batch_size': 4,  # 长序列需要更小的batch size
        'learning_rate': 3e-5,
        'epochs': 200,
        'gradient_accumulation_steps': 4,  # 梯度累积
    }
}

# 多模态情感分析配置
EMOTION_CONFIG = {
    **BASE_CONFIG,
    'model': {
        **BASE_CONFIG['model'],
        'num_classes': 7,  # 7种基本情感
        'hidden_dim': 256,
    },
    'data': {
        **BASE_CONFIG['data'],
        'seq_len': 20,
        'img_dim': 512,   # 面部特征
        'audio_dim': 256, # 更丰富的音频特征
    },
    'training': {
        **BASE_CONFIG['training'],
        'batch_size': 20,
        'learning_rate': 1e-4,
        'epochs': 60,
    }
}

# 视频摘要配置
VIDEO_SUMMARY_CONFIG = {
    **BASE_CONFIG,
    'model': {
        **BASE_CONFIG['model'],
        'num_classes': 2,  # 重要/不重要
        'causal': False,
    },
    'data': {
        **BASE_CONFIG['data'],
        'seq_len': 100,  # 很长的序列
        'img_dim': 1024,
    },
    'training': {
        **BASE_CONFIG['training'],
        'batch_size': 2,
        'learning_rate': 1e-5,
        'epochs': 50,
    }
}

# 配置字典
CONFIGS = {
    'base': BASE_CONFIG,
    'kinetics400': KINETICS400_CONFIG,
    'ucf101': UCF101_CONFIG,
    'realtime': REALTIME_CONFIG,
    'long_video': LONG_VIDEO_CONFIG,
    'emotion': EMOTION_CONFIG,
    'video_summary': VIDEO_SUMMARY_CONFIG,
}


def get_config(config_name='base'):
    """获取指定配置"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name].copy()


def print_config(config_name='base'):
    """打印配置信息"""
    config = get_config(config_name)
    
    print(f"Configuration: {config_name}")
    print("=" * 50)
    
    for section, params in config.items():
        print(f"\n{section.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # 打印所有配置
    for config_name in CONFIGS.keys():
        print_config(config_name)
        print("\n" + "="*80 + "\n")
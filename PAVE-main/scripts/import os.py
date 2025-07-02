import os
import subprocess
import sys

# 强制设置GPU环境
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print(f"设置CUDA_VISIBLE_DEVICES为: {os.environ['CUDA_VISIBLE_DEVICES']}")

# 构建命令
cmd = [
    sys.executable,  # python路径
    '/mnt/sda/shenhao/code/AVQA/PAVE-main/train_pave_w_feat.py',
    '--lora_enable', 'True',
    '--annotation_path', '/mnt/sda/shenhao/code/AVQA/PAVE-main/annots/music_avqa/music_avqa_train_instruct_duplicate_audio.json',
    '--fast_path_mapping_path', '/mnt/sda/shenhao/code/AVQA/PAVE-main/annots/music_avqa/music_avqa_all_imagebind_feature_mapping.json',
    '--slow_path_mapping_path', '/mnt/sda/shenhao/code/AVQA/PAVE-main/annots/music_avqa/music_avqa_all_videos_mapping.json',
    '--data_root', '/mnt/sda/shenhao/datasets/PAVE/audio_imagebind_feat',
    '--slow_path_data_root', '/mnt/sda/shenhao/datasets/MUSIC-AVQA',
    '--use_fast_feat', 'True',
    '--use_slow', 'True',
    '--cache_dir', '/mnt/sda/shenhao/cache/huggingface',
    '--model_name_or_path', 'lmms-lab/llava-onevision-qwen2-0.5b-ov',
    '--version', 'conv_llava_ov_qwen',
    '--model_class', 'VideoFeatModelArgumentsV5_1_3_audio_languagebind_3layers',
    '--output_dir', './checkpoints/pave_v5_1_3_lora_music_avqa0.5B_2epoch_imagebind_3layers',
    '--num_train_epochs', '2',
    '--per_device_train_batch_size', '1',
    '--per_device_eval_batch_size', '1',
    '--gradient_accumulation_steps', '16',
    '--evaluation_strategy', 'no',
    '--save_strategy', 'steps',
    '--save_steps', '2000',
    '--save_total_limit', '1',
    '--learning_rate', '2e-5',
    '--weight_decay', '0.',
    '--warmup_ratio', '0.03',
    '--lr_scheduler_type', 'cosine',
    '--logging_steps', '1',
    '--model_max_length', '2048',
    '--gradient_checkpointing', 'True',
    '--dataloader_num_workers', '2',
    '--lazy_preprocess', 'True',
    '--report_to', 'none',
    '--bf16', 'True',
    '--tf32', 'True',
    '--mm_newline_position', 'grid',
    '--mm_spatial_pool_mode', 'bilinear',
    '--feat_combine_method', 'add',
    '--fast_feat_type', 'audio'
]

# 执行命令
subprocess.run(cmd, env=os.environ)

"""
QA-TIGER配置文件 - 双向对齐策略

该配置启用了双向对齐策略，自动选择最优的对齐方向。
适用于需要灵活对齐策略的复杂场景。
"""

config = dict(
    type='qa-tiger',
    seed=3407,
    epochs=30,
    num_labels=42,
    log_interval=100,
    output_dir='/mnt/sda/shenhao/code/AVQA/QA-TIGER/qa-tiger_clip_vitl14@336px_alignment_bidirectional',
    weight='/mnt/sda/shenhao/code/AVQA/QA-TIGER/qa-tiger_clip_vitl14@336px/2025-06-18-12-34-16_seed713/best.pt',
    pretrained_weight="base",
    mccd=dict(
        flag=False,
        bias_learner=dict(
            three_bias_learner_exist=True,
            q_bias=True,
            a_bias=True,
            v_bias=True,
        ),
        loss_weight=dict(
            major_loss_weight=1,
            distribution_loss_weight=0.1,
            euclidean_distance_fusion_q_weight=0.2,
            euclidean_distance_fusion_a_weight=0.4,
            euclidean_distance_fusion_v_weight=0.4,
            cycle_Kl_loss_weight=0.333,
            cycle_KL_a_q_weight=0.233,
            cycle_KL_q_v_weight=0.233,
            cycle_KL_v_a_weight=0.533,
        ),
        mlp=dict(
            input_dim=512,
            dimensions=[512, 256, 42],
        )
    ),

    data=dict(
        root='./data',
        img_size=336,
        batch_size=28,  # 稍小的batch size，因为双向对齐计算量更大
        eval_batch_size=28,
        num_workers=8,
        frame_sample_rate=1,
        audios_dir='/mnt/sda/shenhao/datasets/MUSIC-AVQA/audio',
        frames_dir='/mnt/sda/shenhao/datasets/MUSIC-AVQA/frames',
        train_annot='./annots/music_avqa/music_avqa_train.json',
        valid_annot='./annots/music_avqa/music_avqa_val.json',
        test_annot='./annots/music_avqa/music_avqa_test.json',
        test_annots=None,
        ans_quelen='./annots/music_avqa/answer2idx.json',

        # precomputed features
        quest_feat=None,
        audio_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/imagebind/audio60',
        video_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/clip_feat/',
        patch_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/tome_feat',
        prompt_feat=None,
    ),

    hyper_params=dict(
        gpus='1',
        model_type="QA-TIGER_ViTL14@336px_Alignment_Bidirectional",
        model=dict(
            d_model=512,
            video_dim=768,
            patch_dim=1024,
            quest_dim=512,
            audio_dim=1024,
            topK=7,
            num_experts=7,
            encoder_type='openai/clip-vit-large-patch14',
            mccd_flag=False,
            use_ams=False,
            use_mamba=False,
            lambda_multifaceted=0.001,
            mamba_config=dict(
                mamba_hidden_dim=256,
                depths=[2],
                num_heads=[16],
                layer_scale=1e-6,
                causal=False,
                question_fusion='concat',
                drop_path_rate=0.1,
            ),
            mamba_aggregator_config=dict(
                d_model=512,
                mamba_hidden_dim=256,
                depths=[2],
                num_heads=[16],
                question_fusion='concat',
                dropout=0.1,
            ),
            
            # 对齐配置 - 双向策略
            use_alignment=True,  # 启用对齐功能
            alignment_config=dict(
                enabled=True,  # 启用对齐
                strategy='bidirectional',  # 双向对齐策略：自动选择最优方向
                lambda_align=0.12,  # 中等对齐损失权重
                ot_eps=1e-8,  # 最优传输数值稳定性参数
                mmd_sigma=1.1,  # 中等MMD核参数
                patch_alignment=True,  # 启用patch特征对齐
                debug_mode=True,  # 启用调试模式以监控方向选择
                memory_efficient=True,  # 启用内存优化，双向计算需要更多内存
            )
        ),
        optim=dict(
            lr=1.6e-4,  # 中等学习率
            encoder_lr=None,
            min_lr=1e-7,
            weight_decay=1e-2,
            betas=(0.95, 0.999)
        ),
        sched=dict(
            name='StepLR',
            mode='min',
            gamma=0.1,
            step_size=8,
            factor=0.5,
            patience=5,
            verbose=True,
            warmup_epochs=2,
        ),
    )
)
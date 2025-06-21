config = dict(
    type='qa-tiger',
    seed=713,
    epochs=15,
    num_labels=42,
    log_interval=100,
    output_dir='/mnt/sda/shenhao/code/AVQA/QA-TIGER/qa-tiger_clip_vitl14@336px',
    weight='/mnt/sda/shenhao/code/AVQA/QA-TIGER/qa-tiger_clip_vitl14@336px/2025-06-18-12-34-16_seed713/best.pt',
    pretrained_weight="base",
    mccd=dict(
        flag=True,
        batch_size=64,
        eval_batch_size=128,
        train_num_workers=8,
        eval_num_workers=8,
        bias_learner=dict(
            three_bias_learner_exist=True,
            q_bias=True,
            a_bias=True,
            v_bias=True,
        ),
        loss_weight=dict(
            major_loss_weight=1,
            distribution_loss_weight=0.01,     # 0.01
            euclidean_distance_fusion_q_weight=0,
            euclidean_distance_fusion_a_weight=0.5,
            euclidean_distance_fusion_v_weight=0.5,
            cycle_Kl_loss_weight=0.333,
            cycle_KL_a_q_weight=0,
            cycle_KL_q_v_weight=0,
            cycle_KL_v_a_weight=0.333,
        ),
        mlp=dict(
            input_dim=512,
            dimensions=[ 512, 256, 42 ],
        )


    ),

    data=dict(
        root='./data',
        img_size=336,
        batch_size=16,
        eval_batch_size=32,
        num_workers=16,
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
        audio_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/audit_feat/60vggish/',
        video_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/clip_feat/',
        patch_feat='/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/tome_feat/',
        prompt_feat=None,
    ),

    hyper_params=dict(
        gpus='1',
        model_type="QA-TIGER_ViTL14@336px",
        model=dict(
            d_model=512,
            video_dim=768,
            patch_dim=1024,
            quest_dim=512,
            audio_dim=128,
            topK=7,
            num_experts=7,
            encoder_type='ViT-L/14@336px',
            mccd_flag=True,
            lambda_multifaceted=0.001,
        ),
        optim=dict(
            lr=1e-4,
            encoder_lr=None,
            min_lr=1e-7,
            weight_decay=0,
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

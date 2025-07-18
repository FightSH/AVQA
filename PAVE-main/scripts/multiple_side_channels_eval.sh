CUDA_VISIBLE_DEVICES=0 python eval_pave_avsd_multi_sides.py \
    --model-path ./checkpoints/pave_v5_1_3_lora_avsd_audio_and_dense_7B  \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_multi_side_audio_languagebind_7B \
    --annotation-file ./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json \
    --video-folder ./data/video_instruction_tuning/avsd/Charades_vu17_test \
    --video-feature-folder ./data/video_instruction_tuning/avsd/Charades_vu17_test_languagebind_feat \
    --audio-feature-folder ./data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat \
    --pred-save ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avsd_audio_and_dense_7B.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen

python tools/audio/avsd/run_coco_eval.py \
--gt-file ./data/video_instruction_tuning/avsd/coco_version_test_gt.json \
--results-file ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avsd_audio_and_dense_7B.json
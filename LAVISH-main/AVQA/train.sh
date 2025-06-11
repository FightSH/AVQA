python net_grd_avst/main_avst.py  --mode train \
	--audio_dir /mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/vggish \
	--video_res14x14_dir /mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/res18_14x14 \
	--wandb 0 --num_workers 32 --batch-size 32 --model_name swinv2_tune_av+vggish
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10001 \
	--use_env /lustre/scratch/client/vinai/users/trangpvh1/repo/FAN/main.py  --data_dir /lustre/scratch/client/vinai/users/trangpvh1/repo/datasets/imagenet/ --model fan_tiny_12_p16_224 -b 128 --sched cosine --epochs 300 \
	--opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
	--model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel \
	--reprob 0.25 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 \
	--drop-path .0 --img-size 224 --mixup 0.8 --cutmix 1.0 \
	--smoothing 0.1 \
	--output lustre/scratch/client/vinai/users/trangpvh1/repo/FAN/output/fan_tiny_12_p16_224/ \
	--amp --model-ema --use_wandb 1 --project_name 'FAN' --job_name sym_baseline

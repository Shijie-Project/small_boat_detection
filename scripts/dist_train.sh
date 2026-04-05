export CUDA_VISIBLE_DEVICES=0,1

torchrun --master_port=7789 --nproc_per_node=2 train.py \
     -c ./configs/dome/Dome-M-custom.yml -t ../ckpts/Dome-M-AITOD-best.pth --seed=42
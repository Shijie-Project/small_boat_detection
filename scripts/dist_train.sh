export CUDA_VISIBLE_DEVICES=0,1

torchrun --master_port=7789 --nproc_per_node=2 train.py \
     -c ./configs/dome/dfine-s-aea.yml --seed=42 \
     2>&1 | tee -a "${outdir}/train.log"
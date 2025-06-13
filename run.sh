CUDA_VISILBE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py \
--cfg ./configs/hsvmambav2v_tiny_224.yaml \
--batch-size 32 \
--data-path /data3/ImageNet1K \
--output ./results
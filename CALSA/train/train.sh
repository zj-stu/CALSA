EXPID=$(date +"%Y%m%d_%H%M%S")
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

HOST='127.0.0.1'
PORT='7'

NUM_GPU=4
python train.py \
--config 'configs/train.yaml' \
--output_dir 'results' \
--checkpoint 'ALBEF_4M.pth' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--model_save_epoch 100 \

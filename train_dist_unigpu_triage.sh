GPU_NUM="1"
CFG="config/cfg_odvg.py"
DATASETS="config/datasets_mixed_odvg_051326.json"
OUTPUT_DIR="/mnt/data/triage_data/models/open-gdino-outputs/vk-person-17k_v1"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /home/ubuntu/airlab_ws/weights/hemobox/models/open_gdino/gdino_swint_darpa-ir-v1-1k_20_1.pth \
        # --options text_encoder_type=/home/ubuntu/airlab_ws/weights/hemobox/models/bert-base-uncased
        --options text_encoder_type=/mnt/data/airlab_ws_full/weights/hemobox/models/bert-base-uncased-v2
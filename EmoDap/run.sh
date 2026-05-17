#!/usr/bin/env bash
# End-to-end EmoDap pipeline (paper Sections 3.2 + 3.3).
#
# 1) Pre-computation:
#    1a) Demonstration Pool Curating  (demo_pool_curating.py)
#    1b) Emotion Prototype Learning   (prototype_training.py)
# 2) Dual-Agent GRPO joint training   (main.py)

set -e

DATASET=${DATASET:-ED}
RUN_SEED=${RUN_SEED:-1}
DATA_ROOT=${DATA_ROOT:-data}
ENCODER=${ENCODER:-sentence-transformers/all-mpnet-base-v2}
LLM_DIR=${LLM_DIR:-/data_server/pengkun/Model/Meta-Llama-3-8B-Instruct}
GPU=${GPU:-0}
K=${K:-8}              # demos per emotion (paper: K=8)
RHO=${RHO:-0.4}        # Top-rho candidate filtering (paper: 0.4)
G=${G:-8}              # GRPO group size  (paper: 8)
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}

BASE_DIR=${DATA_ROOT}/${DATASET}/random_splits2/fold_${RUN_SEED}
PROTO_DIR=ckpt/${DATASET}/proto_fold${RUN_SEED}
DEMO_POOL=${DATA_ROOT}/${DATASET}/uns_candis.json   # produced by step 1a

# ----- 1a) Demonstration Pool Curating --------------------------------------
# Expects a TSV (text<TAB>label) with already self-consistency-pseudo-labelled
# samples from the unlabeled corpus D_u.  See `demo_pool_curating.py` for the
# self-consistency annotation helper.
if [ ! -f "${DEMO_POOL}" ]; then
  echo "[1a] Building demonstration pool ..."
  python demo_pool_curating.py \
    --pseudo_csv ${BASE_DIR}/pseudo_unlabeled.csv \
    --label_json ${BASE_DIR}/test_label.json \
    --encoder ${ENCODER} \
    --output ${DEMO_POOL} \
    --K ${K}
fi

# ----- 1b) Emotion Prototype Learning ---------------------------------------
echo "[1b] Training emotion prototypes ..."
python prototype_training.py \
  --train_csv ${BASE_DIR}/train_see.csv \
  --seen_label_json ${BASE_DIR}/see_relation.json \
  --test_label_json ${BASE_DIR}/test_label.json \
  --demo_pool_json ${DEMO_POOL} \
  --encoder ${ENCODER} \
  --output_dir ${PROTO_DIR} \
  --epochs 5 --tau 0.05 --batch_size 16 --lr 2e-5 \
  --gpu ${GPU}

# ----- 2) Dual-Agent GRPO joint training ------------------------------------
echo "[2] GRPO joint training ..."
python main.py \
  --dataset_name ${DATASET} \
  --run_seed ${RUN_SEED} \
  --data_root ${DATA_ROOT} \
  --demo_pool_json ${DEMO_POOL} \
  --proto_dir ${PROTO_DIR} \
  --llm_dir ${LLM_DIR} \
  --encoder ${ENCODER} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --rho ${RHO} \
  --G ${G} \
  --gpu ${GPU} \
  --save_path ckpt/${DATASET}/emodap_retriever_fold${RUN_SEED}.pt

#!/bin/bash
# Replicate fine-tuning results from the mT5 paper.

export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

#ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=v3-256 --tpu-only --noconf

TASK=mt5_xnli_zeroshot

PRETRAINED_DIR=../mt5large
PRETRAINED_STEPS=1000000
MODEL_DIR="${BUCKET}${TASK}"

# Default MAX_DECODE_LENGTH
# Note: Set it to a larger value when task targets have more than 128 tokens.
MAX_DECODE_LENGTH=128
BATCH_SIZE=1048576
SEQUENCE_LENGTH_GIN="xnli"
FINTUNE_STEPS=8000
# ==== Run fine-tuning ====
CUDA_VISIABLE_DEVICES=0,1 python -m t5.models.mesh_transformer_main \

  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_file="sequence_lengths/${SEQUENCE_LENGTH_GIN}.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-256'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="multilingual_t5.tasks" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', ${FLAGS_BATCH_SIZE})" \
  --eval_gin_param="Bitransformer.decode.max_decode_length = ${FLAGS_MAX_DECODE_LENGTH}" \
  --finetune_steps=${FLAGS_FINTUNE_STEPS}

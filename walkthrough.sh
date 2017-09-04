#!/usr/bin/env bash

# Walkthrough training a good English-to-German translation model using the
# Transformer model from Attention Is All You Need on WMT data.

PROBLEM="translate_ende_wmt32k"
MODEL="transformer"
HPARAMS="transformer_base_single_gpu"

DATA_DIR="${DATA}/t2t_data"
TRAIN_DIR="${DATA}/t2t_train/${PROBLEM}/${MODEL}-${HPARAMS}"
TMP_DIR="/tmp/t2t_datagen"

mkdir -p "${DATA_DIR}" "${TMP_DIR}" "${TRAIN_DIR}"

# ===============================================
# Generate Data.
# ===============================================
t2t-datagen \
  --data_dir="${DATA_DIR}" \
  --tmp_dir="${TMP_DIR}" \
  --problem="${PROBLEM}"


# ===============================================
# Train.
# If run out of memory, add --hparams='batch_size=1024'.
# ===============================================
t2t-trainer \
  --data_dir="${DATA_DIR}" \
  --problems="${PROBLEM}" \
  --model="${MODEL}" \
  --hparams_set="${HPARAMS}" \
  --hparams="batch_size=1024" \
  --output_dir="${TRAIN_DIR}"


# ===============================================
# Decode.
# ===============================================
DECODE_FILE="$DATA_DIR/decode_this.txt"
BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_beam_size=$BEAM_SIZE \
  --decode_alpha=$ALPHA \
  --decode_from_file=$DECODE_FILE

cat $DECODE_FILE.$MODEL.$HPARAMS.beam$BEAM_SIZE.alpha$ALPHA.decodes

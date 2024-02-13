#!/usr/bin/env sh

export PYTHONPATH=..

export TF_CPP_MIN_LOG_LEVEL=3

export CORTEX_LIB_LOG_LEVEL=5

# Один из самых эффективных размеров батча: 16.
options="\
--model-path ../models/captcha_solving/CCS-MCv2 \
--dataset-path ../../minecraft_captcha \
--model-name CCS-MCv2 \
--image-width 128 \
--image-height 128 \
--batch-size 16 \
--epochs-count 350 \
--overtrain true \
--early-stopping-patience 5 \
--dataset-size-multiplier 0.8 \
--shuffle true
"

command="python3.10 cli/captcha_solving_train_cli.py $*${options}"

echo "${command}"
eval "${command}"


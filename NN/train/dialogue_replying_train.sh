#!/usr/bin/env sh

export PYTHONPATH=..

export TF_CPP_MIN_LOG_LEVEL=3

export CORTEX_LIB_LOG_LEVEL=5

options="\
--model-path ../models/dialogue_replying/CDR-T \
--dataset-file ../../cdr_data_v3.json \
--model-name CDR-T \
--batch-size 32 \
--epochs-count 50 \
--overtrain true \
--early-stopping-patience 5 \
--dataset-size-multiplier 0.8 \
--shuffle true
"

command="python cli/dialogue_replying_train_cli.py $*${options}"

echo "${command}"
eval "${command}"


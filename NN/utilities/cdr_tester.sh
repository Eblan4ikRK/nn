#!/usr/bin/env sh

export PYTHONPATH=..

export CORTEX_LIB_LOG_LEVEL=5

options="\
--file-path ../../cdr_arith_data.test.json \
--cdr-url http://0.0.0.0:8080/dialogue_replying/arith \
--tests-count 100
"

command="python cli/cdr_tester_cli.py $*${options}"

echo "${command}"
eval "${command}"

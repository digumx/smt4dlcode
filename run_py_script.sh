#!/usr/bin/env bash

# A simple bash script to run python scripts with automated saving of stdout and stderr. The stdout
# is saved to ./out.log and ./logs/$(date).log

python -u "$@" > >(tee >(tee print.log > "logs/log_$(date).log")) 2>&1

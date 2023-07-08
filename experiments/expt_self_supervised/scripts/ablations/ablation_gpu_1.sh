#!/bin/bash

# models: 4 6
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/
#bash train_pooling.sh obj_000004
#bash train_centroid.sh obj_000004
bash train_pooling.sh obj_000006
bash train_centroid.sh obj_000006

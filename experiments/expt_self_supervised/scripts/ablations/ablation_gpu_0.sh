#!/bin/bash

# models 17 18
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/
#bash train_pooling.sh obj_000017
#bash train_centroid.sh obj_000017
bash train_pooling.sh obj_000018
bash train_centroid.sh obj_000018

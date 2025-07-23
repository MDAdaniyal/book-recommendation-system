#!/bin/bash

# Arguments:
# $1 = train file
# $2 = test file
# $3 = output file

./PPMd_static e $1 model.bin
./PPMd_static d $2 model.bin > $3

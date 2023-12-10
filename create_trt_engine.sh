#! /bin/bash
ONNX="onnx/FastestDet.onnx"

# change this path to your tensorrt trtexec path
TRT_BIN="/home/lizhang/TensorRT-8.5.2.2/bin/trtexec"
# TRT_BIN="/usr/src/tensorrt/bin/trtexec"

TRT="engine/FastDet.engine"


# VERBOSE="--verbose"
VERBOSE=""
 
$TRT_BIN --onnx=$UltraFace_ONNX --saveEngine=$UltraFace_TRT


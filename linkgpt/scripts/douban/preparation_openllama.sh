#!/bin/bash

DATASET_NAME=amazon_clothing_20k
LINKGPT_DATA_PATH=../../data
PROJECT_PATH=../..

# Create directories
mkdir -p ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}
mkdir -p ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}
mkdir -p ${LINKGPT_DATA_PATH}/results/${DATASET_NAME}

echo "Directories created for ${DATASET_NAME} with OpenLLaMA model"
echo "Data path: ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}"
echo "Model path: ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}"
echo "Results path: ${LINKGPT_DATA_PATH}/results/${DATASET_NAME}"

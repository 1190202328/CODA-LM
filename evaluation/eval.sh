#!/bin/bash

#bash /nfs/volume-902-16/tangwenbo/s3_all.sh

model=$2
prediction_dir=$3
ref_name=$4

root=/Users/didi/Desktop/CODA-LM/evaluation
reference_dir=/Users/didi/Desktop/ECCV比赛/验证数据保存/${ref_name}
save_dir=/Users/didi/Downloads/Z_WorkShop
OPENAI_KEY=$1

echo "---step0: convert---"
cd "${root}" && CUDA_VISIBLE_DEVICES=0 python \
  convert2eval.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/region_perception_answer.jsonl"

echo "---step1: eval General perception---"
cd "${root}" && CUDA_VISIBLE_DEVICES=0 python \
  stage1_eval_batch.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/general_perception_answer.jsonl" --save_path "${save_dir}/${ref_name}/general_perception_answer" --model_name "${model}" --api_key "${OPENAI_KEY}"

echo "---step2: eval Driving suggestion---"
cd "${root}" && CUDA_VISIBLE_DEVICES=0 python \
  stage2_eval_batch.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/driving_suggestion_answer.jsonl" --save_path "${save_dir}/${ref_name}/driving_suggestion_answer" --model_name "${model}" --api_key "${OPENAI_KEY}"

echo "---step3: eval Regional perception---"
cd "${root}" && CUDA_VISIBLE_DEVICES=0 python \
  stage3_eval_batch.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/region_perception_answer_w_label.jsonl" --save_path "${save_dir}/${ref_name}/region_perception_answer" --model_name "${model}" --api_key "${OPENAI_KEY}"

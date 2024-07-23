#!/bin/bash

# 脚本参数
OPENAI_KEY=$1
# 定义其他变量
model=$2
prediction_dir=$3
ref_name=$4
time_gap=$5
shift

root=/Users/didi/Desktop/CODA-LM/evaluation
reference_dir=/Users/didi/Desktop/ECCV比赛/验证数据保存/${ref_name}
save_dir=/Users/didi/Downloads/Z_WorkShop

# 检查是否提供了步骤参数
if [ $# -eq 0 ]; then
  echo "Error: No steps specified."
  exit 1
fi

echo "---step0: convert---"
cd "${root}" && CUDA_VISIBLE_DEVICES=0 python convert2eval.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/region_perception_answer.jsonl"

# 循环遍历所有步骤参数
for step in "$@"; do
  case "$step" in
  step1)
    echo "---step1: eval General perception---"
    cd "${root}" && CUDA_VISIBLE_DEVICES=0 python stage1_eval_batch.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/general_perception_answer.jsonl" --save_path "${save_dir}/${ref_name}/general_perception_answer" --model_name "${model}" --api_key "${OPENAI_KEY}" --time_gap "${time_gap}"
    ;;
  step2)
    echo "---step2: eval Driving suggestion---"
    cd "${root}" && CUDA_VISIBLE_DEVICES=0 python stage2_eval_batch.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/driving_suggestion_answer.jsonl" --save_path "${save_dir}/${ref_name}/driving_suggestion_answer" --model_name "${model}" --api_key "${OPENAI_KEY}" --time_gap "${time_gap}"
    ;;
  step3)
    echo "---step3: eval Regional perception---"
    cd "${root}" && CUDA_VISIBLE_DEVICES=0 python stage3_eval_batch.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/region_perception_answer_w_label.jsonl" --save_path "${save_dir}/${ref_name}/region_perception_answer" --model_name "${model}" --api_key "${OPENAI_KEY}" --time_gap "${time_gap}"
    ;;
  esac
done

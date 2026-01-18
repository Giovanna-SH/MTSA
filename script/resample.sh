# ==========================================
# 1. 环境与路径配置
# ==========================================
# 加载 CUDA 环境
module load cuda-12.1

# 1.1 设置模型路径
# [注意] 红队模型路径：沿用你步骤3中动态获取的方式，或者直接指定具体路径
export Red_team_model_path=$(ls -d $(pwd)/model_output/red_team_model_data20260114231131/checkpoint-597 | head -1)

# [注意] 目标模型路径 (Zephyr-7b-beta)
export Target_model_path=/home/s202510023/workspace/MTSA/models/zephyr-7b-beta

# [注意] 裁判模型路径 (Llama-Guard-3)
export Llama_Guard_3_8B=/home/s202510023/workspace/MTSA/models/Llama-Guard-3-8B

# [新增] 相似度模型路径 (all-MiniLM-L6-v2)
# 请务必确认这个路径是否存在！如果你的模型在不同位置，请修改此处。
export Sim_model_name=/home/s202510023/workspace/MTSA/models/all-MiniLM-L6-v2

# 1.2 输入/输出文件配置
# 步骤3生成的攻击结果文件
export INPUT_ATTACK_RESULTS=attack_results/adversarial_generate.json
# 步骤4将要生成的重采样输出目录
export OUTPUT_DIR=attack_results/red_team_dpo
# 最终输出文件路径
export OUTPUT_FILE=${OUTPUT_DIR}/red_team_tragectory_resample.json

# ==========================================
# 2. 准备工作
# ==========================================
# 创建输出目录 (防止报错 FileNotFoundError)
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_ATTACK_RESULTS" ]; then
    echo "❌ 错误: 找不到步骤3的输出文件: $INPUT_ATTACK_RESULTS"
    echo "请检查该文件是否已生成。"
    exit 1
fi

# ==========================================
# 3. 创建后台运行脚本 (run_resample.sh)
# ==========================================
cat > run_resample.sh << EOF
#!/bin/bash
module load cuda-12.1

echo ">>> 任务开始: 红队轨迹重采样 (Red Team Trajectory Resample)"
echo ">>> 攻击者 (Red Team): \$Red_team_model_path"
echo ">>> 相似度模型 (Sim):  \$Sim_model_name"
echo ">>> 输入文件: $INPUT_ATTACK_RESULTS"

# 运行重采样代码
# 注意：此脚本通常只需要单卡即可，这里指定 cuda:0
CUDA_VISIBLE_DEVICES=0,1 python -u red_team_tragectory_resample.py \\
    --attack_model_name "\$Red_team_model_path" \\
    --judge_model_name "\$Llama_Guard_3_8B" \\
    --target_model_name "\$Target_model_path" \\
    --sim_model_name "\$Sim_model_name" \\
    --attack_gpu cuda:0 \\
    --target_gpu cuda:1 \\
    --judge_gpu cuda:1 \\
    --sim_gpu cuda:0 \\
    --attack_results $INPUT_ATTACK_RESULTS \\
    --output_dir $OUTPUT_FILE

echo ">>> 任务结束！结果已保存至: $OUTPUT_FILE"
EOF

# ==========================================
# 4. 启动任务
# ==========================================
# 后台启动，日志写入 resample.log
nohup bash run_resample.sh > resample.log 2>&1 &

echo "✅ 重采样任务已在后台启动！"
echo "🔎 请输入 'tail -f resample.log' 查看实时进度。"
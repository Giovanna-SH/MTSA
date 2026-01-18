# ==========================================
# 1. 恢复环境配置 (必做！)
# ==========================================
# 加载 CUDA
module load cuda-12.1

# 重新设置裁判模型路径 (Q)

export Red_team_model_path=$(ls -d $(pwd)/model_output/red_team_model_data20260114231131/checkpoint-597 | head -1)
# 重新设置裁判模型路径 (Llama-Guard-3)
export Llama_Guard_3_8B=/home/s202510023/workspace/MTSA/models/Llama-Guard-3-8B

# 重新设置目标模型路径 (Zephyr-7b)
export Target_model_path=/home/s202510023/workspace/MTSA/models/zephyr-7b-beta


# ==========================================
# 2. 准备工作
# ==========================================
# 创建输出目录 (否则脚本会报 FileNotFoundError)
mkdir -p attack_results

# ==========================================
# 3. 创建后台运行脚本 (run_gen.sh)
# ==========================================
cat > run_gen.sh << EOF
#!/bin/bash
module load cuda-12.1

echo ">>> 任务开始: 对抗生成 (Adversarial Generate)"
echo ">>> 攻击者 (Red Team): \$Red_team_model_path"
echo ">>> 目标 (Target): \$Target_model_path"
echo ">>> 裁判 (Judge): \$Llama_Guard_3_8B"
CUDA_VISIBLE_DEVICES=0,1 python -u adversarial_generate.py  \\
        --attack_model_name "\$Red_team_model_path" \\
        --judge_model_name "\$Llama_Guard_3_8B" \\
        --target_model_name "\$Target_model_path" \\
        --attack_gpu cuda:0 \\
        --target_gpu cuda:1 \\
        --judge_gpu cuda:1 \\
        --attack_data_path datasets/attack_target/train_attack_target.json \\
        --output_dir attack_results/adversarial_generate.json

echo ">>> 任务结束！"
EOF

# ==========================================
# 4. 启动任务
# ==========================================
# 后台启动，日志写入 gen.log
nohup bash run_gen.sh > gen.log 2>&1 &

echo "✅ 对抗生成任务已在后台启动！"
echo "🔎 请输入 'tail -f gen.log' 查看实时进度。"
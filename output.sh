#!/bin/bash

# 1. 출력 폴더 생성 (없으면 생성)
OUTPUT_DIR="output_policy"
mkdir -p "$OUTPUT_DIR"

# 2. results 폴더 내의 각 ablation 폴더를 순회
for ablation_dir in results/toddlerbot__T_Walk_ppo_*_ablation; do
    if [ -d "$ablation_dir" ]; then
        ablation_name=$(basename "$ablation_dir")
        echo "=== Ablation: $ablation_name ==="

        # * 부분 추출: toddlerbot__T_Walk_ppo_{*}_ablation 에서 * 부분을 꺼내고 _ 제거
        wildcard_part=$(echo "$ablation_name" | sed 's/^toddlerbot__T_Walk_ppo_//;s/_ablation$//' | tr -d '_')

        # ablation 폴더명에서 True/False 추출
        if echo "$ablation_name" | grep -q "enabled=True"; then
            thermal_suffix="true"
        else
            thermal_suffix="false"
        fi

        # 각 ablation 폴더 내의 실험 폴더를 순회
        for policy_dir in "$ablation_dir"/*; do
            if [ -d "$policy_dir" ]; then
                policy_name=$(basename "$policy_dir")
                echo "Processing: $policy_name"

                # best_policy 파일을 {policy_name}_{true/false} 로 복사
                if [ -f "$policy_dir/best_policy" ]; then
                    cp "$policy_dir/best_policy" "${OUTPUT_DIR}/${policy_name}_${wildcard_part}"
                    echo "  - Copied best_policy -> ${policy_name}_${wildcard_part}"
                fi
            fi
        done
    fi
done

echo "---"
echo "작업 완료! 파일이 '$OUTPUT_DIR' 폴더에 정리되었습니다."
#!/bin/bash

# 1. 출력 폴더 생성 (없으면 생성)
OUTPUT_DIR="output_policy"
mkdir -p "$OUTPUT_DIR"

# 2. results 폴더 내의 각 정책 폴더를 순회
# results/toddlerbot__T_Walk_ppo_ablation/* 로 정책 폴더 리스트를 가져옵니다.
for policy_dir in results/toddlerbot__T_Walk_ppo_ablation/*; do
    
    # 디렉토리인지 확인
    if [ -d "$policy_dir" ]; then
        # 경로에서 폴더명만 추출 (예: results/policy_A -> policy_A)
        policy_name=$(basename "$policy_dir")
        
        echo "Processing: $policy_name"

        # (1) best_policy 파일 처리
        if [ -f "$policy_dir/best_policy" ]; then
            cp "$policy_dir/best_policy" "${OUTPUT_DIR}/${policy_name}_best_policy"
            echo "  - Copied best_policy"
        fi

        # (2) 511488000/policy 파일 처리
        if [ -f "$policy_dir/511488000/policy" ]; then
            cp "$policy_dir/511488000/policy" "${OUTPUT_DIR}/${policy_name}_511488000_policy"
            echo "  - Copied 511488000"
        fi
    fi
done

echo "---"
echo "작업 완료! 파일이 '$OUTPUT_DIR' 폴더에 정리되었습니다."
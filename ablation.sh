#!/bin/bash

# 1. 결과 저장 폴더 생성
mkdir -p ./ablation

# 2. 결과 폴더 내의 model_*_2026* 패턴 탐색
for dir in ./results/toddlerbot__T_Walk_ppo_ablation/model_*_2026*/ ; do
    
    # 폴더 존재 확인
    [ -d "$dir" ] || continue

    # 폴더명 추출 (예: model_a_f_20260206)
    dirname=$(basename "$dir")

    # 3. 이름 변경: 마지막 '_' 이후(날짜 부분)를 통째로 삭제
    # 결과 예시: model_a_f_20260206 -> model_a_f
    new_name="${dirname%_*}"

    # 원본 파일 경로
    src_file="${dir}best_policy"

    # 파일 복사 (best_policy 파일이 존재할 때만)
    if [ -f "$src_file" ]; then
        cp "$src_file" "./ablation/$new_name"
        echo "성공: $dirname/best_policy -> ./ablation/$new_name"
    else
        echo "실패: $dirname 내에 best_policy 파일이 없습니다."
    fi
done

echo "모든 복사 작업이 완료되었습니다."
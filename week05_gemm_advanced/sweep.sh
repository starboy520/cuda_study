#!/usr/bin/env bash
# Day 5 参数实验：扫 TM/TN/BM/BN/BK，记录 寄存器数 + GFLOPS + 正确性
# 用法：
#   ./sweep.sh            # 默认在 2048 上扫预设组合
#   ./sweep.sh 1024       # 指定规模
set -u

SRC=gemm_2d_thread_tiling.cu
ARCH=sm_80
SIZE=${1:-2048}
TMPBIN=/tmp/gemm_sweep

printf "%-22s %-5s %-10s %-6s\n" "TMxTN BMxBN BK" "reg" "GFLOPS" "chk"
printf -- "------------------------------------------------\n"

sweep() {
  local TM=$1 TN=$2 BM=$3 BN=$4 BK=$5
  local build reg out g c
  build=$(nvcc -arch=$ARCH -O3 -Xptxas -v \
          -DTM=$TM -DTN=$TN -DBM=$BM -DBN=$BN -DBK=$BK \
          "$SRC" -o "$TMPBIN" 2>&1)
  if [ $? -ne 0 ]; then
    printf "%-22s %-5s %-10s %-6s\n" "${TM}x${TN} ${BM}x${BN} ${BK}" "-" "-" "BUILD"
    return
  fi
  reg=$(echo "$build" | grep -oP '\d+(?= registers)')
  out=$("$TMPBIN" "$SIZE" 2>&1)
  g=$(echo "$out" | grep cost | tail -1 | grep -oP '[0-9.]+(?= GFLOPS)')
  c=$(echo "$out" | grep -oE 'PASS|FAIL')
  [ -z "$c" ] && c="ERR"
  printf "%-22s %-5s %-10s %-6s\n" "${TM}x${TN} ${BM}x${BN} ${BK}" "${reg:--}" "${g:--}" "$c"
}

echo "# GEMM 参数扫描  (arch=$ARCH, size=$SIZE)"

#      TM TN  BM  BN  BK
sweep   8  8  64  64  32     # 现状基线
sweep   8  8  64  64  64     # 降寄存器
#sweep   8  8  64  64  16     # 只改 BK
#sweep   4  4  64  64  16     # 降寄存器
#sweep   8  4  64  64  16     # 折中
#sweep   8  8 128  64  16     # 大 tile（行方向）
#sweep   8  8 128 128  16     # 更大 tile
#sweep   4  4 128 128  16     # 大 tile + 小 TM/TN

rm -f "$TMPBIN"

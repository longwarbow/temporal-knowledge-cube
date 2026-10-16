#!/usr/bin/env bash
# E15 聚类批驱动：charliehebdo 深做（全链），其余 6 事件浅做（--l2-only）
cd "$(dirname "$0")" || exit 1
PY="D:/WSPforWB/10.keyan/.venv_e14/Scripts/python.exe"

step () {
  local tag="$1"; local log="$2"; shift 2
  echo "=== $tag ==="
  "$PY" -u e15_p1_cluster_evolve.py "$@" > "$log" 2>&1
  echo "exit=$?"
  tail -8 "$log"
  echo
}

step ch _e15_p1_cl_ch.log --event charliehebdo-all-rnr-threads   --slices 6
step fe _e15_p1_cl_fe.log --event ferguson-all-rnr-threads       --slices 4 --l2-only
step ot _e15_p1_cl_ot.log --event ottawashooting-all-rnr-threads --slices 4 --l2-only
step pr _e15_p1_cl_pr.log --event prince-toronto-all-rnr-threads --slices 4 --l2-only --kmin 8 --kmax 14
step pu _e15_p1_cl_pu.log --event putinmissing-all-rnr-threads   --slices 4 --l2-only --kmin 8 --kmax 14
step eb _e15_p1_cl_eb.log --event ebola-essien-all-rnr-threads   --slices 3 --l2-only --kmin 4 --kmax 8
step gu _e15_p1_cl_gu.log --event gurlitt-all-rnr-threads        --slices 4 --l2-only --kmin 8 --kmax 14

echo "ALL_CLUSTER_DONE"

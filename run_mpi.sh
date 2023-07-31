#!/bin/bash

# 最小値と最大値を設定
min_np=1
max_np=300

# ログファイルを設定
logfile="nbody_log_10.log"

# ログファイルが存在すれば削除
if [ -f $logfile ]; then
    rm $logfile
fi

# 最小値から最大値までループ
for np in $(seq $min_np 10 $max_np)
do
    echo "Running with np = $np" >> $logfile
    mpirun -np $np ./nbody_mpi 3000 1000 0.001 100 >> $logfile
    echo "---------------------------" >> $logfile
done


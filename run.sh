#!/bin/bash

# プログラム名とパラメーターを変数に格納
PROGRAM="./nbody"
PARAMETERS="3000 1000 0.001 100"

# 出力するログファイル名
LOGFILE="nbody_times.log"

# ログファイルが存在する場合は削除
if [ -e $LOGFILE ]; then
   rm $LOGFILE
fi

# OMP_NUM_THREADSを1から56まで変化させてプログラムを実行
for i in $(seq 1 56); do
   export OMP_NUM_THREADS=$i
   echo "Running with OMP_NUM_THREADS=$i" | tee -a $LOGFILE
   $PROGRAM $PARAMETERS | tee -a $LOGFILE
done


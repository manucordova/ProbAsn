#! /bin/bash

i0=$1
n=$2
i="0"

for f in `ls ../../../ProbAsn_ShiftML2/data/ShiftML2_probasn_dataset/`
do
  i=$((i+1))

  if [[ "$i" -ge "$i0" && "$i" -lt "$((i0+n))" ]]; then
    echo $f
    python Make_DB_ShiftML2.py $f
  fi

done

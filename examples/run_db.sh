#! /bin/bash

for f in `ls ../../Data/Additional_dataset/`
do

  if ! grep -Fq $f done.txt
  then

    echo $f >> done.txt
    python Add_to_DB.py $f

  fi

  if grep -Fq "yes" stop.txt
  then
    exit 0
  fi

done

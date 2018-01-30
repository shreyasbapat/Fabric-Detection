#!/bin/bash
read a
counter=1
cd Fas/$a
for f in $(ls -1)
do
  new_filename=$(printf "%d_.jpg" ${counter})     
  echo "renaming ${f} ..to.. ${new_filename}"
  mv ${f} ${new_filename}
  (( counter=${counter}+1 ))
done

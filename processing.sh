#!/bin/bash

a=1
b=10
for i in {0..49}
do
    d=`expr $a + $i`
    begin=`expr $i \* $b`
    end=`expr $d \* $b`
    echo $begin
    echo $end
    nohup python big_computation.py frechet $begin $end > log/_big_computation_$i.log &
done
#!/usr/bin/env bash
n=100
cc=4
maxRound=100
sigma_w=0
sigma_z=0
sigma_b=0
sigma_a=0
#0 0.01 0.02 0.03 0.04 
#0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06
#for sigma_b in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2;
for n in 30 50 40 60; do
    for cc in 1 2 3 4; do
        for tier in 0 1 2 3 4; do
            echo $n $cc $tier
            for x in {1..10}; do
                python2.7 doOneRun.py ts $maxRound $n $cc $sigma_w $sigma_z $sigma_b $sigma_a new tier $tier
            done;
        done;
    done;
done;




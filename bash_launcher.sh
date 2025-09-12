#!/usr/bin/env bash
n=100
cc=4
maxRound=100
sigma_w=0
sigma_z=0
sigma_b=0
sigma_a=0
tier=0
#0 0.01 0.02 0.03 0.04 
#0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06
#for sigma_b in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2;
for x in {1..5}; do
    for n in 200 250 300; do
        for cc in 2 2; do
        #for tier in 0 1 2 3 4; do
            echo $n $cc $x
            ~/miniforge3/envs/rewiring/bin/python3.10 run.py --exp-type ts --nb-rounds $maxRound --nb-firms $n --cc $cc --sigma-w $sigma_w --sigma-z $sigma_z --sigma-b $sigma_b --sigma-a $sigma_a --aisi-spread 0.01 --network-type new --exp-name slowing --tier $tier
        #done;
        done;
    done;
done;




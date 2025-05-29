#!/bin/bash
#SBATCH -J MonteCarlo
#SBATCH -A cs475-575
#SBATCH -p classgputest
#SBATCH --gres=gpu:1
#SBATCH -o montecarlo.out
#SBATCH -e montecarlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@oregonstate.edu
for t in 1024 4096 16384 65536 262144 1048576 2097152
do
    for b in 8 32 64 128 256
    do
        /usr/local/apps/cuda/11.7/bin/nvcc -DNUMTRIALS=$t -DBLOCKSIZE=$b -DCSV -o proj05 proj05.cu
        ./proj05 >> results.csv
    done
done

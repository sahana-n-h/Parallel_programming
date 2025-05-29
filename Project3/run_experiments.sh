#!/bin/bash

# Add header to CSV file
echo -e "NUMT\tNUMCITIES\tNUMCAPITALS\tmegaCityCapitalsPerSecond" > results.csv

# Iterate over threads and number of capitals
for t in 1 2 4 6 8
do
  for n in 2 3 4 5 10 15 20 30 40 50
  do
    # Set OpenMP threads and compile code with the current NUMT and NUMCAPITALS values
    export OMP_NUM_THREADS=$t
    clang++ -DNUMT=$t -DNUMCAPITALS=$n -Xpreprocessor -fopenmp \
        -I/opt/homebrew/opt/libomp/include \
        -L/opt/homebrew/opt/libomp/lib -lomp \
        Proj03.cpp -o Proj03

    # Check if compilation was successful
    if [ $? -ne 0 ]; then
      echo "Compilation failed for NUMT=$t, NUMCAPITALS=$n" >&2
      continue
    fi

    # Run the compiled program and save only the CSV result
    ./Proj03 >> results.csv
  done
done
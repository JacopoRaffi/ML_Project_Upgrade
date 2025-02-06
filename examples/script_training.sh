#!/bin/bash

g++ -c train_example.cpp -o train_example.o
g++ train_example.o ../build/*.o -o train_example

./train_example 

#gnuplot -p -e "plot 'loss.txt' using 1:2 with lines title 'Train Loss', 'loss.txt' using 1:3 with lines title 'Test Loss'"

rm train_example.o train_example loss.txt
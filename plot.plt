set term qt
set output
set xlabel "Iteration #"
set ylabel "Value"
set title 'results.txt'
set key autotitle columnhead
plot for [i=2:10] "results.txt" using 1:i  with linespoints
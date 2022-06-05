set term qt
set output
set xlabel "Iteration #"
set ylabel "Value"
set yrange [0:1]
set title 'std_results.txt'
set key autotitle columnhead
plot for [i=2:20:2] "std_results.txt" using 1:i:($1+i) with yerrorbars
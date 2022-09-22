set style data lines
set xrange [0:50]
set yrange [0:1.0]
set xlabel "iteration"
set ylabel "strategies freq"
#set key at 95,80
#set label "b=1.4, k=6" at 5,110
#set label "num of experiments=50" at 200,110
#set label "b=1.4" at 200,105
plot 'std_results.txt' using 1:8:9 with yerrorbars lc 7 title "stdev: all-C",\
'std_results.txt' using 1:8 with lines lc 7 lw 3 title "avg num all-C",\
'std_results.txt' using 1:10:11 with yerrorbars lc 6 title "stdev: all-D",\
'std_results.txt' using 1:10 with lines lc 6 lw 3 title "avg num of all-D",\
'std_results.txt' using 1:12:13 with yerrorbars lc 2 title "stdev: k-D",\
'std_results.txt' using 1:12 with lines lc 2 lw 3 title "avg num of k-D",\
'std_results.txt' using 1:14:15 with yerrorbars lc 3 title "stdev: k-C",\
'std_results.txt' using 1:14 with lines lc 3 lw 3 title "avg num of k-C",\
'std_results.txt' using 1:16:17 with yerrorbars lc 9 title "stdev: k-DC",\
'std_results.txt' using 1:16 with lines lc 9 lw 3 title "avg num of k-DC"

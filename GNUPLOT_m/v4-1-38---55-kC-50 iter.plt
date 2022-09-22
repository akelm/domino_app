set style data lines
set xrange [0:50]
set yrange [0:1.0]
set xlabel "iteration"
set ylabel "kD freq"
#set key at 95,80
#set label "b=1.4, k=6" at 5,110
#set label "num of experiments=50" at 200,110
#set label "b=1.4" at 200,105
plot 'std_results.txt' using 1:38:39 with yerrorbars lc 7 title "stdev: 0-C",\
'std_results.txt' using 1:38 with lines lc 7 lw 3 title "av 0-C",\
'std_results.txt' using 1:40:41 with yerrorbars lc 3 title "stdev: 1-C",\
'std_results.txt' using 1:40 with lines lc 3 lw 3 title "av 1-C",\
'std_results.txt' using 1:42:43 with yerrorbars lc 8 title "stdev: 2-C",\
'std_results.txt' using 1:42 with lines lc 8 lw 3 title "av 2-C",\
'std_results.txt' using 1:44:45 with yerrorbars lc 2 title "stdev: 3-C",\
'std_results.txt' using 1:44 with lines lc 2 lw 3 title "av 3-C",\
'std_results.txt' using 1:46:47 with yerrorbars lc 4 title "stdev: 4-C",\
'std_results.txt' using 1:46 with lines lc 4 lw 3 title "av 4-C",\
'std_results.txt' using 1:48:49 with yerrorbars lc 5 title "stdev: 5-C",\
'std_results.txt' using 1:48 with lines lc 5 lw 3 title "av 5-C",\
'std_results.txt' using 1:50:51 with yerrorbars lc 6 title "stdev: 6-C",\
'std_results.txt' using 1:50 with lines lc 6 lw 3 title "av 6-C",\
'std_results.txt' using 1:52:53 with yerrorbars lc 9 title "stdev: 7-C",\
'std_results.txt' using 1:52 with lines lc 9 lw 3 title "av 7-C",\
'std_results.txt' using 1:54:55 with yerrorbars lc 4 title "stdev: 8-C",\
'std_results.txt' using 1:54 with lines lc 4 lw 3 title "av 8-C"





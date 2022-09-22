set style data lines
set xrange [0:50]
set yrange [0:1.0]
set xlabel "iteration"
set ylabel "kD freq"
#set key at 95,80
#set label "b=1.4, k=6" at 5,110
#set label "num of experiments=50" at 200,110
#set label "b=1.4" at 200,105
plot 'std_results.txt' using 1:20:21 with yerrorbars lc 7 title "stdev: 0-D",\
'std_results.txt' using 1:20 with lines lc 7 lw 3 title "av 0-D",\
'std_results.txt' using 1:22:23 with yerrorbars lc 3 title "stdev: 1-D",\
'std_results.txt' using 1:22 with lines lc 3 lw 3 title "av 1-D",\
'std_results.txt' using 1:24:25 with yerrorbars lc 8 title "stdev: 2-D",\
'std_results.txt' using 1:24 with lines lc 8 lw 3 title "av 2-D",\
'std_results.txt' using 1:26:27 with yerrorbars lc 2 title "stdev: 3-D",\
'std_results.txt' using 1:26 with lines lc 2 lw 3 title "av 3-D",\
'std_results.txt' using 1:28:29 with yerrorbars lc 4 title "stdev: 4-D",\
'std_results.txt' using 1:28 with lines lc 4 lw 3 title "av 4-D",\
'std_results.txt' using 1:30:31 with yerrorbars lc 5 title "stdev: 5-D",\
'std_results.txt' using 1:30 with lines lc 5 lw 3 title "av 5-D",\
'std_results.txt' using 1:32:33 with yerrorbars lc 6 title "stdev: 6-D",\
'std_results.txt' using 1:32 with lines lc 6 lw 3 title "av 6-D",\
'std_results.txt' using 1:34:35 with yerrorbars lc 9 title "stdev: 7-D",\
'std_results.txt' using 1:34 with lines lc 9 lw 3 title "av 7-D",\
'std_results.txt' using 1:36:37 with yerrorbars lc 4 title "stdev: 8-D",\
'std_results.txt' using 1:36 with lines lc 4 lw 3 title "av 8-D"





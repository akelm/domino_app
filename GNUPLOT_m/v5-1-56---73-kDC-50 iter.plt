set style data lines
set xrange [0:50]
set yrange [0:1.0]
set xlabel "iteration"
set ylabel "kD freq"
#set key at 95,80
#set label "b=1.4, k=6" at 5,110
#set label "num of experiments=50" at 200,110
#set label "b=1.4" at 200,105
plot 'std_results.txt' using 1:56:57 with yerrorbars lc 7 title "stdev: 0-DC",\
'std_results.txt' using 1:56 with lines lc 7 lw 3 title "av 0-DC",\
'std_results.txt' using 1:58:59 with yerrorbars lc 3 title "stdev: 1-DC",\
'std_results.txt' using 1:58 with lines lc 3 lw 3 title "av 1-DC",\
'std_results.txt' using 1:60:61 with yerrorbars lc 8 title "stdev: 2-DC",\
'std_results.txt' using 1:60 with lines lc 8 lw 3 title "av 2-DC",\
'std_results.txt' using 1:62:63 with yerrorbars lc 2 title "stdev: 3-DC",\
'std_results.txt' using 1:62 with lines lc 2 lw 3 title "av 3-DC",\
'std_results.txt' using 1:64:65 with yerrorbars lc 4 title "stdev: 4-DC",\
'std_results.txt' using 1:64 with lines lc 4 lw 3 title "av 4-DC",\
'std_results.txt' using 1:66:67 with yerrorbars lc 5 title "stdev: 5-DC",\
'std_results.txt' using 1:66 with lines lc 5 lw 3 title "av 5-DC",\
'std_results.txt' using 1:68:69 with yerrorbars lc 6 title "stdev: 6-DC",\
'std_results.txt' using 1:68 with lines lc 6 lw 3 title "av 6-DC",\
'std_results.txt' using 1:70:71 with yerrorbars lc 9 title "stdev: 7-DC",\
'std_results.txt' using 1:70 with lines lc 9 lw 3 title "av 7-DC",\
'std_results.txt' using 1:72:73 with yerrorbars lc 4 title "stdev: 8-DC",\
'std_results.txt' using 1:72 with lines lc 4 lw 3 title "av 8-DC"





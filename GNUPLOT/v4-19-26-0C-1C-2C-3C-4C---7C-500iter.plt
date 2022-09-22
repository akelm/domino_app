set style data lines
set xrange [0:500]
set yrange [-0.1:1.3]
set xlabel "iteration"
set ylabel "freq kC"
#set key at 95,80
#set label "b=1.4, k=4" at 5,110
#set label "num of experiments=50" at 200,110
#set label "50 single runs" at 50,110
#set label "b=1.4" at 50,100
plot 'results.txt' using 1:20 with lines lc 7 lw 3 title "freq 0-C",\
 'results.txt' using 1:21 with lines lc 3 lw 3  title "freq 1-C",\
 'results.txt' using 1:22 with lines lc 8 lw 3 title "freq 2-C",\
 'results.txt' using 1:23 with lines lc 2 lw 3 title "freq 3-C",\
'results.txt' using 1:24 with lines lc 4 lw 3 title "freq 4-C",\
'results.txt' using 1:25 with lines lc 5 lw 3  title "freq 5-C",\
 'results.txt' using 1:26 with lines lc 6 lw 3 title "freq 6-C",\
 'results.txt' using 1:27 with lines lc 9 lw 3 title "freq 7-C",\
'results.txt' using 1:28 with lines lc 4 lw 3 title "freq 8-C"    
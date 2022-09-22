set style data lines
set xrange [0:500]
set yrange [-0.1:1.3]
set xlabel "iteration"
set ylabel "freq kDC"
#set key at 95,80
#set label "b=1.4, k=4" at 5,110
#set label "num of experiments=50" at 200,110
#set label "50 single runs" at 50,110
#set label "b=1.4" at 50,100
plot 'results.txt' using 1:29 with lines lc 7 lw 3 title "freq 0-DC",\
 'results.txt' using 1:30 with lines lc 3 lw 3  title "freq 1-DC",\
 'results.txt' using 1:31 with lines lc 8 lw 3 title "freq 2-DC",\
 'results.txt' using 1:32 with lines lc 2 lw 3 title "freq 3-DC",\
'results.txt' using 1:33 with lines lc 4 lw 3 title "freq 4-DC",\
'results.txt' using 1:34 with lines lc 5 lw 3  title "freq 5-DC",\
 'results.txt' using 1:35 with lines lc 6 lw 3 title "freq 6-DC",\
 'results.txt' using 1:36 with lines lc 9 lw 3 title "freq 7-DC",\
'results.txt' using 1:37 with lines lc 4 lw 3 title "freq 8-DC"   
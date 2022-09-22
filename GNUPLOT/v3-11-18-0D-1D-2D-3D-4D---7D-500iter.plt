set style data lines
set xrange [0:500]
set yrange [-0.1:1.3]
set xlabel "iteration"
set ylabel "freq kD"
#set key at 95,80
#set label "b=1.4, k=4" at 5,110
#set label "num of experiments=50" at 200,110
#set label "50 single runs" at 50,110
#set label "b=1.4" at 50,100
plot 'results.txt' using 1:11 with lines lc 7 lw 3 title "freq 0-D",\
 'results.txt' using 1:12 with lines lc 3 lw 3  title "freq 1-D",\
 'results.txt' using 1:13 with lines lc 8 lw 3 title "freq 2-D",\
 'results.txt' using 1:14 with lines lc 2 lw 3 title "freq 3-D",\
'results.txt' using 1:15 with lines lc 4 lw 3 title "freq 4-D",\
'results.txt' using 1:16 with lines lc 5 lw 3  title "freq 5-D",\
 'results.txt' using 1:17 with lines lc 6 lw 3 title "freq 6-D",\
 'results.txt' using 1:18 with lines lc 9 lw 3 title "freq 7-D",\
'results.txt' using 1:18 with lines lc 4 lw 3 title "freq 8-D"  
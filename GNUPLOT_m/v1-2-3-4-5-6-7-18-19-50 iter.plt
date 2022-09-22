set style data lines
set xrange [0:50]
set yrange [-0.5:1.1]
set xlabel "iteration"
set ylabel "f_C, C-corr, av payoff"
#set key at 95,80
#set label "b=1.4, k=6" at 5,110
#set label "num of experiments=50" at 200,110
#set label "b=1.4" at 200,105
plot 'std_results.txt' using 1:2:3 with yerrorbars lc 7 title "stdev: av f C",\
 'std_results.txt' using 1:2 with lines lc 7 lw 3 title "av f C",\
'std_results.txt' using 1:4:5 with yerrorbars lc 3 title "stdev: av f C corr",\
'std_results.txt' using 1:4 with lines lc 3 lw 3 title "av f C corr",\
'std_results.txt' using 1:6:7 with yerrorbars lc 4 title "stdev: av av sum",\
'std_results.txt' using 1:6 with lines lc 4 lw 3 title "av sum",\
'std_results.txt' using 1:18:19 with yerrorbars lc 1 title "stdev: av str change",\
'std_results.txt' using 1:18 with lines lc 1 lw 3 title "av str change"




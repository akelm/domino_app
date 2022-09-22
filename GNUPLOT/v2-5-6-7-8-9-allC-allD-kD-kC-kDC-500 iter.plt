set style data lines
set xrange [0:500]
set yrange [0:1.0]
set xlabel "iteration"
set ylabel "agents [%]"
#set key at 95,80
#set label "b=1.4, k=4" at 5,110
#set label "num of experiments=50" at 200,110
#set label "50 single runs" at 50,110
#set label "b=1.4" at 50,100
plot 'results.txt' using 1:5 with lines lc 7 lw 3 title "cells with All-C",\
 'results.txt' using 1:6 with lines lc 6 lw 3  title "cells with All-D",\
 'results.txt' using 1:7 with lines lc 2 lw 3 title "cells with k-D",\
'results.txt' using 1:8 with lines lc 3 lw 3 title "cells with k-C",\
'results.txt' using 1:9 with lines lc 1 lw 3 title "cells with k-DC"

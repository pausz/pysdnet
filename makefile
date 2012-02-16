sdde_step:
	gcc -shared -fPIC -lm -o sdde_step.so -std=c99 sdde_step.c


movie:
	mencoder "mf://*.png" -mf type=png:fps=3 -ovc lavc -o output.avi


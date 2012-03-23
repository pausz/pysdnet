sdde_step:
	gcc -shared -fPIC -lm -o csdde.so -O3 -std=c99 csdde.c

movie:
	mencoder "mf://*.png" -mf type=png:fps=3 -ovc lavc -o output.avi


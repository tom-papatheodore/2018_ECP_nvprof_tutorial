CUCOMP  = nvcc
CUFLAGS = -arch=sm_60 -DUSE_DOUBLE
LFLAGS  = -lcublas

run: poisson2d.o
	$(CUCOMP) $(CUFLAGS) $(LFLAGS) poisson2d.o -o run

poisson2d.o: poisson2d.cu common.h
	$(CUCOMP) $(CUFLAGS) -c poisson2d.cu

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall: clean
	rm -f *.nvvp *.prof *.err *.out

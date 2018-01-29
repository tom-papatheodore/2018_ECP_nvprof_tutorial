CUCOMP  = nvcc
CUFLAGS = -arch=sm_60 -DUSE_DOUBLE
INCLUDE = -I$(OMPI_DIR)/include -L$(OMPI_DIR)/lib
LFLAGS  = -lcublas -lmpi_ibm

run: poisson2d.o
	$(CUCOMP) $(INCLUDE) $(CUFLAGS) $(LFLAGS) poisson2d.o -o run

poisson2d.o: poisson2d.cu common.h
	$(CUCOMP) $(CUFLAGS) -c poisson2d.cu

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall: clean
	rm -f *.nvvp *.prof *.err *.out

CUCOMP = nvcc
CUFLAGS = -arch=sm_60

run: vector_addition.o
	$(CUCOMP) $(CUFLAGS) vector_addition.o -o run

vector_addition.o: vector_addition.cu
	$(CUCOMP) $(CUFLAGS) -c vector_addition.cu

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall: clean
	rm -f *.nvvp *.prof *.err *.out

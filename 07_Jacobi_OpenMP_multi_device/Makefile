CCOMP  = pgcc
CFLAGS = -DUSE_DOUBLE -mp -Minfo=accel -fast -acc -ta=tesla:cc60 -Msafeptr

run: poisson2d.o
	$(CCOMP) $(CFLAGS) poisson2d.o -o run

poisson2d.o: poisson2d.c common.h
	$(CCOMP) $(CFLAGS) -c  poisson2d.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall: clean
	rm -f *.nvvp *.prof *.err *.out

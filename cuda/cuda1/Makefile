include ../makefile.vars

NUM=1

%.o: %.cu
	nvcc -arch sm_21 -c $<

all: main$(NUM).cpp cuda$(NUM).o 
	g++ -g $? $(CUDALIBS) $(OPENCVLIBS) -o cuda$(NUM)

clean:
	rm -f *.o cuda$(NUM)



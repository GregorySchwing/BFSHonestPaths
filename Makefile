CC=g++
CXX=$(CC)
CFLAGS=-lm -O3
CXXFLAGS=$(CFLAGS)
LD=$(CXX)
NVCC=nvcc
LDFLAGS = $(CFLAGS)
NVCC=nvcc
CUDAFLAGS = -O3 -Xptxas -O3 -Xcompiler -O3 -w 

SOURCES = bipartite.cu matchgpu.cu CSRGraph.cu GreedyMatcher.cu bfs.cu
OBJECTS = bipartite.o matchgpu.o CSRGraph.o GreedyMatcher.o bfs.o

%.o: %.cu
	${NVCC} -dc $(CUDAFLAGS) $< -o $@

bfshonest_lib.a: ${OBJECTS}
	$(NVCC) -lib *.o -o bfshonest_lib.a $(LDFLAGS)

clean :
	${RM} *.o *.a Makefile.bak core
	
lint :
	${LINT} ${SOURCES}

depend :
	makedepend ${SOURCES}

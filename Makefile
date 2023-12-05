CC=g++
CXX=$(CC)
CFLAGS=-lm -O3
CXXFLAGS=$(CFLAGS)
LD=$(CXX)
NVCC=nvcc
LDFLAGS = $(CFLAGS)
NVCC=nvcc
CUDAFLAGS = -O3 -Xptxas -O3 -Xcompiler -O3 -w 


all: bfshonest_lib.a

bfshonest_lib.a: bipartite.cu matchgpu.cu CSRGraph.cu GreedyMatcher.cu bfs.cu
	$(NVCC) $(LDFLAGS) $(CUDAFLAGS)  -c -o bipartite.o bipartite.cu
	$(NVCC) $(LDFLAGS) $(CUDAFLAGS)  -c -o matchgpu.o matchgpu.cu
	$(NVCC) $(LDFLAGS) $(CUDAFLAGS)  -c -o CSRGraph.o CSRGraph.cu
	$(NVCC) $(LDFLAGS) $(CUDAFLAGS)  -c -o GreedyMatcher.o GreedyMatcher.cu
	$(NVCC) $(LDFLAGS) $(CUDAFLAGS)	 -c -o bfs.o bfs.cu
	$(NVCC) -lib *.o -o bfshonest_lib.a $(LDFLAGS)

clean: 
	-rm -rf *.o *.a bfshonest_lib
depend:
	makedepend -Y *.cu *.c *.hpp

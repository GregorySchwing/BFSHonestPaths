#ifndef CSRGRAPH_CUH
#define CSRGRAPH_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/sort.h>

template <typename T>
__global__ void setNumInArray(T *arrays, T *index, T *value, int num_index);

struct CSRGraph {
public:
    CSRGraph(int _n, int _m, int * rows, int * cols, int * matching);
    void createOffsets();
    void copyMatchingBack();

//private:
    const int INF = 1e9;
    unsigned int m;
    unsigned int n;
    int *rows, *cols, *matching;
    thrust::device_vector<unsigned int> rows_d;
    thrust::device_vector<unsigned int> cols_d;
    thrust::device_vector<char> vals_d;
    thrust::device_vector<unsigned int> offsets_d;
    thrust::device_vector<unsigned int> keylabel_d;
    thrust::device_vector<unsigned int> nonzerodegrees_d;
    thrust::device_vector<unsigned int> degrees_d;
    thrust::device_vector<int> mate_d;
};

#endif

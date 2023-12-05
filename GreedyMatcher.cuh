#ifndef GREEDY_MATCHER_CUH
#define GREEDY_MATCHER_CUH

#include "CSRGraph.cuh"
#include "matchgpu.h"
#define THREADS_PER_BLOCK 1024

struct is_less_than_0
{
  __host__ __device__ int operator()(int &x);
};

struct GreedyMatcher
{
  GreedyMatcher(CSRGraph &_csr);
  int maxMatch();

  CSRGraph &csr;
  thrust::device_vector<int> c_h;
  thrust::device_vector<int> req_d;
  thrust::device_vector<int> c_d;
  int *c_Pinned;
};

#endif

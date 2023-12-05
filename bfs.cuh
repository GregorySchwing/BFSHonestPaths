#ifndef BFS_CUH
#define BFS_CUH

#include "CSRGraph.cuh"
#include "GreedyMatcher.cuh"

namespace BFSHonestPaths {

struct BFS
{
  BFS(CSRGraph &_csr, GreedyMatcher &_gm);

  int augmentNaivePaths(int num_iterations);

  CSRGraph &csr;
  GreedyMatcher &gm;

  thrust::host_vector<unsigned int> f_h;
  thrust::host_vector<unsigned int> ft_h;
  thrust::host_vector<int> S_h;
  thrust::host_vector<int> pred_h;
  thrust::host_vector<float> sigma_h;
  thrust::host_vector<int> search_tree_src_h;
  thrust::host_vector<int> c_h;

  thrust::host_vector<unsigned int> BTypePair_list_counter_h;
  thrust::host_vector<uint64_t> BTypePair_list_h;
  thrust::host_vector<unsigned int> BTypePair_disjoint_list_counter_h;
  thrust::host_vector<uint64_t> BTypePair_disjoint_list_h;

  thrust::device_vector<unsigned int> f_d;
  thrust::device_vector<unsigned int> ft_d;
  thrust::device_vector<int> S_d;
  thrust::device_vector<int> pred_d;
  thrust::device_vector<float> sigma_d;
  thrust::device_vector<int> search_tree_src_d;
  thrust::device_vector<int> c_d;

  thrust::device_vector<unsigned int> BTypePair_list_counter_d;
  thrust::device_vector<uint64_t> BTypePair_list_d;
  thrust::device_vector<unsigned int> BTypePair_disjoint_list_counter_d;
  thrust::device_vector<uint64_t> BTypePair_disjoint_list_d;
  thrust::device_vector<int> m2_d;
};

}
#endif  // BFS_CUH

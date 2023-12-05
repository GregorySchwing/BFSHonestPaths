
#include "GreedyMatcher.cuh"

namespace BFSHonestPaths {

int is_less_than_0::operator()(int &x)
{
  if (x > -1)
    return x;
  else
    return -1;
}



GreedyMatcher::GreedyMatcher(CSRGraph &_csr) : csr(_csr)
{
  c_h.resize(1, 1);
  req_d.resize(csr.n, 0);
  c_d.resize(1, 1);
  cudaMallocHost((void**)&c_Pinned, sizeof(int)); // host pinned
}

int GreedyMatcher::maxMatch()
{
  unsigned int *rows_d_ptr = thrust::raw_pointer_cast(csr.offsets_d.data());
  unsigned int *cols_d_ptr = thrust::raw_pointer_cast(csr.cols_d.data());
  //int *m_d_ptr = thrust::raw_pointer_cast(m_d.data());
  int *m_d_ptr = thrust::raw_pointer_cast(csr.mate_d.data());
  int *req_d_ptr = thrust::raw_pointer_cast(req_d.data());
  int *c_ptr = thrust::raw_pointer_cast(c_d.data());
  srand(1);
  /*computing MM */
  int matchround = 0;
  int dimGrid = (csr.n + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
  memset(c_Pinned, 1, sizeof(int));
  while (c_Pinned[0] && ++matchround < NR_MAX_MATCH_ROUNDS)
  {
    //printf("match round %d\n", matchround);
    //c_h[0] = 0;
    //c_d = c_h;
    memset(c_Pinned, 0, sizeof(int));
    gaSelect<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, c_ptr, csr.n, rand());
    grRequest<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, req_d_ptr, m_d_ptr, csr.n);
    grRespond<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, req_d_ptr, m_d_ptr, csr.n);
    gMatch<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, req_d_ptr, csr.n);
    cudaMemcpy(c_Pinned, thrust::raw_pointer_cast(c_d.data()), sizeof(int), cudaMemcpyDeviceToHost);
    //c_h = c_d;
  }
  using namespace thrust::placeholders;
  thrust::for_each(csr.mate_d.begin(), csr.mate_d.end(), _1 -= 4);
  thrust::transform(csr.mate_d.begin(), csr.mate_d.end(), csr.mate_d.begin(), is_less_than_0()); // in-place transformation
  //m_h = csr.mate_d;
  //for (int i = 0; i < m_h.size(); ++i)
  //    printf("%d %d\n", i, m_h[i]);
  //for (int i = 0; i < m_h.size(); ++i)
  //  mate[i] = m_h[i];
  int numAugmented = thrust::count_if(csr.mate_d.begin(), csr.mate_d.end(), _1 > -1);
  return numAugmented / 2;
}



}
#include "CSRGraph.cuh"

// kernel function
template <typename T>
__global__ void setNumInArray(T *arrays, T *index, T *value, int num_index)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= num_index || index[tid] < tid)
    return;
  arrays[index[tid]] = value[tid];
}

CSRGraph::CSRGraph(int _n, int _m, int * _rows, int * _cols, int * _matching) 
{

  m = _m;
  n = _n;

  rows = _rows;
  cols = _cols;
  matching = _matching;

  rows_d.resize(2 * m);
  cols_d.resize(2 * m);
  vals_d.resize(2 * m, 1);
  mate_d.resize(n,0);

  thrust::copy(rows, rows + (2*m), rows_d.begin());
  thrust::copy(cols, cols + (2*m), cols_d.begin());

  offsets_d.resize(n + 1);

  keylabel_d.resize(n);
  nonzerodegrees_d.resize(n);
  // This will be the degrees array.
  degrees_d.resize(n);

  createOffsets();
}

void CSRGraph::createOffsets()
  {
    thrust::sort_by_key(thrust::device, rows_d.begin(), rows_d.end(), cols_d.begin());
    thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator> new_end;
    new_end = thrust::reduce_by_key(thrust::device, rows_d.begin(), rows_d.end(), vals_d.begin(), keylabel_d.begin(), nonzerodegrees_d.begin());
    int block_size = 64;
    int num_blocks = (n + block_size - 1) / block_size;
    unsigned int *degrees_ptr_d = thrust::raw_pointer_cast(degrees_d.data());
    unsigned int *keylabel_ptr_d = thrust::raw_pointer_cast(keylabel_d.data());
    unsigned int *nonzerodegrees_ptr_d = thrust::raw_pointer_cast(nonzerodegrees_d.data());
    setNumInArray<unsigned int><<<num_blocks, block_size>>>(degrees_ptr_d, keylabel_ptr_d, nonzerodegrees_ptr_d, n);
    thrust::inclusive_scan(thrust::device, degrees_d.begin(), degrees_d.end(), offsets_d.begin() + 1); // in-place scan

    keylabel_d.clear();
    vals_d.clear();
    nonzerodegrees_d.clear();
  }


void CSRGraph::copyMatchingBack()
{
  thrust::copy(mate_d.begin(), mate_d.end(), matching);
}
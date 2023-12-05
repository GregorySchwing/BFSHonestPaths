#include "CSRGraph.cuh"

CSRGraph::CSRGraph(int _n, int _nnz, int * _rows, int * _cols, int * _matching) 
{

  m = _nnz;
  n = _n;

  rows = _rows;
  cols = _cols;
  matching = _matching;

  offsets_d.resize(n+1);
  cols_d.resize(m);
  mate_d.resize(n,0);

  thrust::copy(rows, rows + (n+1), offsets_d.begin());
  thrust::copy(cols, cols + m, cols_d.begin());

}

void CSRGraph::copyMatchingBack()
{
  thrust::copy(mate_d.begin(), mate_d.end(), matching);
}
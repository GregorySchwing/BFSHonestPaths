#include <bits/stdc++.h>

namespace BFSHonestPaths {

__global__ void setSuperSource_no_bloss_simple(unsigned int *CP_d, unsigned int *IC_d, unsigned int *f_d, int *S_d, float *sigma_d, int *m_d, int *search_tree_src_d, int n)
{
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadId >= n)
    return;
  if (m_d[threadId] == -1)
  {
    f_d[threadId] = 1;
    sigma_d[threadId] = 1.0;
    S_d[threadId] = 0;
    search_tree_src_d[threadId] = threadId;
  }
} // end  setSuperSource

__global__ void spMvUgCscScKernel_all_edges(unsigned int *CP_d, unsigned int *IC_d, unsigned int *ft_d, unsigned int *f_d,
                                            float *sigma_d, int *pred_d, int *search_tree_src_d, int d, int n)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {
    // compute spmv
    ft_d[i] = 0;
    if (sigma_d[i] < 0.01)
    {
      int sum = 0;
      int k;
      int start = CP_d[i];
      int end = CP_d[i + 1];
      for (k = start; k < end; k++)
      {
        if (f_d[IC_d[k]])
        {
          pred_d[i] = IC_d[k];
          search_tree_src_d[i] = search_tree_src_d[IC_d[k]];
          sum += f_d[IC_d[k]];
        }
      }
      if (sum > 0.9)
      {
        ft_d[i] = sum;
      }
    }
  }
} // end spMvUgCscScKernel

__global__ void spMvUgCscScKernel_matched_edge(int *m_d, unsigned int *ft_d, unsigned int *f_d,
                                               float *sigma_d, int *pred_d, int *search_tree_src_d, int d, int n)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {
    // compute spmv
    ft_d[i] = 0;
    if (sigma_d[i] < 0.01)
    {
      int sum = 0;
      if (m_d[i] > -1)
      {
        if (f_d[m_d[i]])
        {
          pred_d[i] = m_d[i];
          search_tree_src_d[i] = search_tree_src_d[m_d[i]];
          sum += f_d[m_d[i]];
        }
      }
      if (sum > 0.9)
      {
        ft_d[i] = sum;
      }
    }
  }
} // end spMvUgCscScKernel

/**************************************************************************/
/*
 * assign vector ft_d to vector f_d,
 * check that the vector f_d  has at least one nonzero element
 * add the vector f to vector sigma.
 * compute the S vector.
 */
__global__ void bfsFunctionsKernel(unsigned int *f_d, unsigned int *ft_d, float *sigma_d, int *S_d, int *c,
                                   int n, int d)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {
    f_d[i] = 0;
    if (ft_d[i] > 0.9)
    {
      *c = 1;
      f_d[i] = ft_d[i];
      sigma_d[i] += ft_d[i];
      S_d[i] = d;
    }
  }
} // end  bfsFunctionsKernel

__global__ void setAllPaths_augmenting(unsigned int *CP_d, unsigned int *IC_d, int *m_d, int *S_in_d, float *sigma_d, int *search_tree_src_d, int n, uint64_t *BTypePair_list_d, unsigned int *BTypePair_list_counter_d)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n)
  {

    // printf("%d %d %d %d %f\n", i,S_in_d[i],search_tree_src_d[i],m_d[i],sigma_d[i]);
    if (sigma_d[i] > 0.1)
    {

      // One of these will be -1.
      int d = S_in_d[i];
      if (d % 2)
      {
        if (m_d[i] > -1)
        {
          int foundB = 1;
          foundB &= S_in_d[i] == S_in_d[m_d[i]];
          foundB &= search_tree_src_d[i] != search_tree_src_d[m_d[i]];
          foundB &= m_d[search_tree_src_d[i]] == -1;
          foundB &= m_d[search_tree_src_d[m_d[i]]] == -1;
          if (foundB)
          {
            // if(0 == atomicCAS(mutex, 0, 1)){
            uint32_t leastSignificantWord = i;
            uint32_t mostSignificantWord = m_d[i];

            uint64_t edgePair = (uint64_t)mostSignificantWord << 32 | leastSignificantWord;
            int topLocal = atomicAdd(BTypePair_list_counter_d, 1);
            if (topLocal >= n)
            {
              //#ifndef NDEBUG
              //printf("EXCEEDED N %u %u, S[%u]= %d, S[%u]= %d; search_tree_src_d[%d]=%d search_tree_src_d[%d]=%d\n", leastSignificantWord, mostSignificantWord, leastSignificantWord, S_in_d[leastSignificantWord], mostSignificantWord, S_in_d[mostSignificantWord], i, search_tree_src_d[i], m_d[i], search_tree_src_d[m_d[i]]);
              //#endif
            }
            else
            {
              BTypePair_list_d[topLocal] = edgePair;
            }
          }
        }
      }
      else
      {
        int k;
        int start = CP_d[i];
        int end = CP_d[i + 1];
        for (k = start; k < end; k++)
        {
          if (sigma_d[IC_d[k]] > 0.0 && S_in_d[i] == S_in_d[IC_d[k]])
          {
            int foundB = S_in_d[i] == S_in_d[IC_d[k]] &&
                         search_tree_src_d[i] != search_tree_src_d[IC_d[k]] &&
                         m_d[search_tree_src_d[i]] == -1 &&
                         m_d[search_tree_src_d[IC_d[k]]] == -1;
            if (foundB)
            {
              // if(0 == atomicCAS(mutex, 0, 1)){
              uint32_t leastSignificantWord = i;
              uint32_t mostSignificantWord = IC_d[k];
              uint64_t edgePair = (uint64_t)mostSignificantWord << 32 | leastSignificantWord;
              int topLocal = atomicAdd(BTypePair_list_counter_d, 1);
              if (topLocal >= n)
              {
                //#ifndef NDEBUG
                //printf("EXCEEDED N %u %u, S[%u]= %d, S[%u]= %d; search_tree_src_d[%d]=%d search_tree_src_d[%d]=%d\n", leastSignificantWord, mostSignificantWord, leastSignificantWord, S_in_d[leastSignificantWord], mostSignificantWord, S_in_d[mostSignificantWord], i, search_tree_src_d[i], m_d[i], search_tree_src_d[m_d[i]]);
                //#endif
              }
              else
              {
                BTypePair_list_d[topLocal] = edgePair;
              }
            }
          }
        }
      }
    }
  }
}


__global__ void lift_path_parallel(int *m_d, int *pred_d, uint64_t *BTypePair_disjoint_list_d, unsigned int *BTypePair_disjoint_list_counter_d)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= BTypePair_disjoint_list_counter_d[0])
    return;
  int curr_v = (int)(BTypePair_disjoint_list_d[i] >> 32);
  int curr_u = (int)(uint32_t)BTypePair_disjoint_list_d[i];

  const int a = curr_u;
  const int b = curr_v;
  int last_a;
  int last_b;
  int a_length = 0;
  int curr_a = a;
  bool matchFirst = m_d[a] == b;
  if (matchFirst)
  {
    while (curr_a != -1)
    {
      last_a = curr_a;
      if (a_length % 2)
      {
        curr_a = pred_d[curr_a];
      }
      else
      {
        curr_a = pred_d[curr_a];
        m_d[curr_a] = last_a;
        m_d[last_a] = curr_a;
      }
      a_length++;
    }
  }
  else
  {
    while (curr_a != -1)
    {
      last_a = curr_a;
      if (a_length % 2)
      {
        curr_a = pred_d[curr_a];
        m_d[curr_a] = last_a;
        m_d[last_a] = curr_a;
      }
      else
      {
        curr_a = pred_d[curr_a];
      }
      a_length++;
    }
  }
  int b_length = 0;
  int curr_b = b;
  if (matchFirst)
  {
    while (curr_b != -1)
    {
      last_b = curr_b;
      if (b_length % 2)
      {
        curr_b = pred_d[curr_b];
      }
      else
      {
        curr_b = pred_d[curr_b];
        m_d[curr_b] = last_b;
        m_d[last_b] = curr_b;
      }
      b_length++;
    }
  }
  else
  {
    while (curr_b != -1)
    {
      last_b = curr_b;
      if (b_length % 2)
      {
        curr_b = pred_d[curr_b];
        m_d[curr_b] = last_b;
        m_d[last_b] = curr_b;
      }
      else
      {
        curr_b = pred_d[curr_b];
      }
      b_length++;
    }
  }
  if (!matchFirst)
  {
    m_d[a] = b;
    m_d[b] = a;
  }
}

}
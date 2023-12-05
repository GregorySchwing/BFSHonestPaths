#include "bfs.cuh"
#include "bfs_kernels.cuh"

namespace BFSHonestPaths {

BFS::BFS(CSRGraph &_csr, GreedyMatcher &_gm) : csr(_csr), gm(_gm)
  {

    f_h.resize(csr.n);
    ft_h.resize(csr.n);
    S_h.resize(csr.n);
    pred_h.resize(csr.n);
    sigma_h.resize(csr.n);
    search_tree_src_h.resize(csr.n, -1);
    c_h.resize(1, 1);

    f_d.resize(csr.n, 0);
    ft_d.resize(csr.n, 0);
    S_d.resize(csr.n, 0);
    pred_d.resize(csr.n, -1);
    sigma_d.resize(csr.n, 0.0);
    search_tree_src_d.resize(csr.n, -1);
    c_d.resize(1, 1);

    BTypePair_list_counter_d.resize(1, 0);
    BTypePair_list_d.resize(csr.n, 0);
    BTypePair_disjoint_list_counter_d.resize(1, 0);
    BTypePair_disjoint_list_d.resize(csr.n, 0);
    m2_d.resize(csr.n, 0);
  }

int BFS::augmentNaivePaths()
  {

    int numAugmented = 0;

    unsigned int *rows_d_ptr = thrust::raw_pointer_cast(csr.offsets_d.data());
    unsigned int *cols_d_ptr = thrust::raw_pointer_cast(csr.cols_d.data());

    unsigned int *f_d_ptr = thrust::raw_pointer_cast(f_d.data());
    unsigned int *ft_d_ptr = thrust::raw_pointer_cast(ft_d.data());

    int *S_d_ptr = thrust::raw_pointer_cast(S_d.data());
    float *sigma_d_ptr = thrust::raw_pointer_cast(sigma_d.data());
    int *search_tree_src_d_ptr = thrust::raw_pointer_cast(search_tree_src_d.data());
    int *pred_d_ptr = thrust::raw_pointer_cast(pred_d.data());

    int *m_d_ptr = thrust::raw_pointer_cast(csr.mate_d.data());
    //int *m_d_ptr = thrust::raw_pointer_cast(gm.m_d.data());
    int *c_d_ptr = thrust::raw_pointer_cast(c_d.data());
    int *c_d_m_ptr = thrust::raw_pointer_cast(gm.c_d.data());

    uint64_t *BTypePair_list_d_ptr = thrust::raw_pointer_cast(BTypePair_list_d.data());
    unsigned int *BTypePair_list_counter_d_ptr = thrust::raw_pointer_cast(BTypePair_list_counter_d.data());
    uint64_t *BTypePair_disjoint_list_d_ptr = thrust::raw_pointer_cast(BTypePair_disjoint_list_d.data());
    unsigned int *BTypePair_disjoint_list_counter_d_ptr = thrust::raw_pointer_cast(BTypePair_disjoint_list_counter_d.data());

    int *m2_d_ptr = thrust::raw_pointer_cast(m2_d.data());
    int *req_d_ptr = thrust::raw_pointer_cast(gm.req_d.data());
    // initialize all of the edges of the super source to
    //  f_d[r] = 1;
    //  sigma_d[r] = 1.0;
    int dimGrid = (csr.n + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;

    do
    {

      setSuperSource_no_bloss_simple<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, f_d_ptr, S_d_ptr, sigma_d_ptr, m_d_ptr, search_tree_src_d_ptr, csr.n);
      int d = 0;
      c_h[0] = 1;
      while (c_h[0])
      {
        d = d + 1;
        c_h[0] = 0;
        c_d = c_h;
        if (d % 2)
        {
          spMvUgCscScKernel_all_edges<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, ft_d_ptr, f_d_ptr, sigma_d_ptr, pred_d_ptr, search_tree_src_d_ptr, d, csr.n);
        }
        else
        {
          spMvUgCscScKernel_matched_edge<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, ft_d_ptr, f_d_ptr, sigma_d_ptr, pred_d_ptr, search_tree_src_d_ptr, d, csr.n);
        }
        bfsFunctionsKernel<<<dimGrid, THREADS_PER_BLOCK>>>(f_d_ptr, ft_d_ptr, sigma_d_ptr, S_d_ptr, c_d_ptr, csr.n, d);
        c_h = c_d;
      }
      setAllPaths_augmenting<<<dimGrid, THREADS_PER_BLOCK>>>(rows_d_ptr, cols_d_ptr, m_d_ptr, S_d_ptr, sigma_d_ptr, search_tree_src_d_ptr, csr.n, BTypePair_list_d_ptr, BTypePair_list_counter_d_ptr);
      BTypePair_list_counter_h = BTypePair_list_counter_d;
      printf("Non-disjoint %d paths\n", BTypePair_list_counter_h[0]);
      cudaMemset(req_d_ptr, 0, sizeof(*req_d_ptr) * csr.n);

      srand(1);
      /*computing MM */
      int matchround = 0;
      gm.c_h[0] = 1;
      while (gm.c_h[0] && ++matchround < NR_MAX_MATCH_ROUNDS)
      {
        // printf("match round %d\n", matchround);
        gm.c_h[0] = 0;
        gm.c_d = gm.c_h;
        gaSelect<<<dimGrid, THREADS_PER_BLOCK>>>(m2_d_ptr, c_d_m_ptr, csr.n, rand());
        grRequestEdgeList<<<dimGrid, THREADS_PER_BLOCK>>>(BTypePair_list_d_ptr, search_tree_src_d_ptr, BTypePair_list_counter_d_ptr, req_d_ptr, m2_d_ptr, csr.n);
        grRespondEdgeList<<<dimGrid, THREADS_PER_BLOCK>>>(BTypePair_list_d_ptr, search_tree_src_d_ptr, BTypePair_list_counter_d_ptr, req_d_ptr, m2_d_ptr, csr.n);
        gMatchEdgeList<<<dimGrid, THREADS_PER_BLOCK>>>(BTypePair_disjoint_list_d_ptr, BTypePair_disjoint_list_counter_d_ptr, BTypePair_list_d_ptr, search_tree_src_d_ptr, BTypePair_list_counter_d_ptr, m2_d_ptr, req_d_ptr, csr.n);
        cudaMemset(req_d_ptr, 0, sizeof(*req_d_ptr) * csr.n);
        gm.c_h = gm.c_d;
      }
      BTypePair_disjoint_list_counter_h = BTypePair_disjoint_list_counter_d;
      printf("Disjoint %d paths\n", BTypePair_disjoint_list_counter_h[0]);

      lift_path_parallel<<<dimGrid, THREADS_PER_BLOCK>>>(m_d_ptr, pred_d_ptr, BTypePair_disjoint_list_d_ptr, BTypePair_disjoint_list_counter_d_ptr);
      numAugmented += BTypePair_disjoint_list_counter_h[0];

      cudaMemset(f_d_ptr, 0, sizeof(*f_d_ptr) * csr.n);
      cudaMemset(ft_d_ptr, 0, sizeof(*f_d_ptr) * csr.n);
      cudaMemset(S_d_ptr, -1, sizeof(*S_d_ptr) * csr.n);

      cudaMemset(sigma_d_ptr, 0, sizeof(*sigma_d_ptr) * csr.n);
      cudaMemset(pred_d_ptr, -1, sizeof(*pred_d_ptr) * csr.n);
      cudaMemset(search_tree_src_d_ptr, -1, sizeof(*search_tree_src_d_ptr) * csr.n);

      cudaMemset(m2_d_ptr, 0, sizeof(*m2_d_ptr) * csr.n);
      cudaMemset(BTypePair_list_counter_d_ptr, 0, sizeof(*BTypePair_list_counter_d_ptr));
      cudaMemset(BTypePair_disjoint_list_counter_d_ptr, 0, sizeof(*BTypePair_disjoint_list_counter_d_ptr));

    } while (BTypePair_disjoint_list_counter_h[0] > 0);
    /*
    gm.m_h = gm.m_d;
    for (int i = 0; i < gm.m_h.size(); ++i)
      mate[i] = gm.m_h[i];
    */
    return numAugmented;
  }

// Implement other member functions here

}
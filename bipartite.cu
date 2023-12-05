/*
 * bipartite.c -- Initialize bipartite graph on GPU
 */

/*
 * Copyright 2023 by Greg Schwing
 */


/*
 * Synopsis
 *
 * This implementation of bipartite matching on the GPU can accelerate general
 * graph matching by initializing nearly the entire matching in parallel.
 *
 */

/*
 * Author
 *
 * Greg Schwing
 * gregory.schwing@med.wayne.edu
 *
 * Department of Computer Science
 * Wayne State University
 * Detroit, MI 48201
 *
 */

/*
 * History
 *
 * 22 October 2023
 * First draft.
 *
 */
#include "bipartite.h"
#include "CSRGraph.cuh"
#include "GreedyMatcher.cuh"
#include "bfs.cuh"

#include <chrono>

void BFSHonestWrapper(int * Rows, int * Cols, int * Matching, int n, int nnz){
    std::chrono::time_point<std::chrono::steady_clock> m_StartTime = std::chrono::steady_clock::now();
    //printf("Generating CSR with %d rows, %d columns\n",m,m);
    CSRGraph csr(n,nnz,Rows,Cols,Matching);
    
    GreedyMatcher gm(csr);
    BFS bfs(csr,gm);
    std::chrono::time_point<std::chrono::steady_clock> m_EndTime = std::chrono::steady_clock::now();
    double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;  
    std::cout << "Creating Greedy Match and BFS datastructures: " << elapsedSeconds << std::endl;

    m_StartTime = std::chrono::steady_clock::now();
    int numAugmented = gm.maxMatch();
    m_EndTime = std::chrono::steady_clock::now();
    elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;  
    std::cout << "Greedy match seconds: " << elapsedSeconds << "; edges augmented: " << numAugmented << std::endl;
    
    m_StartTime = std::chrono::steady_clock::now();
    int numAugmented2 = bfs.augmentNaivePaths();
    m_EndTime = std::chrono::steady_clock::now();
    elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count() / 1000.0;  
    std::cout << "BFS seconds: " << elapsedSeconds << "; edges augmented: " << numAugmented2 << std::endl;
    
    csr.copyMatchingBack();

    //BFS b(csr, gm);
}
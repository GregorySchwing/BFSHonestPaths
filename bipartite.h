
/*
 * biparite.h -- Initialize bipartite graph
 */

/*
 * Copyright 2023 by Greg Schwing
 */


#ifndef Bipartite
#define Bipartite

// *.h file

// ...
#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void bipartite(int * Rows, int * Cols, int * Matching, int n, int m);

#undef EXTERNC
// ...

#endif
/*
Copyright 2011, Bas Fagginger Auer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <exception>
#include <string>
#include <algorithm>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_functions.h>

#include "matchgpu.h"

namespace BFSHonestPaths {

//==== General matching kernels ====
/*
   Match values match[i] have the following interpretation for a vertex i:
   0   = blue,
   1   = red,
   2   = unmatchable (all neighbours of i have been matched),
   3   = reserved,
   >=4 = matched.
*/

//Nothing-up-my-sleeve working constants from SHA-256.
__constant__ const uint dMD5K[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
				0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
				0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
				0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
				0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
				0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
				0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
				0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

//Rotations from MD5.
__constant__ const uint dMD5R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
				5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
				4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
				6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

__constant__ const uint dSelectBarrier = 0x88B81733;

#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

__global__ void gSelect(int *match, int *dkeepMatching, const int nrVertices, const uint random)
{
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	//Can this vertex still be matched?
	//if (match[i] >= 2) return;
	if (match[i] >= 3) return;

	//Start hashing.
	uint h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
	uint a = h0, b = h1, c = h2, d = h3, e, f, g = i;

	for (int j = 0; j < 16; ++j)
	{
		f = (b & c) | ((~b) & d);

		e = d;
		d = c;
		c = b;
		b += LEFTROTATE(a + f + dMD5K[j] + g, dMD5R[j]);
		a = e;

		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;

		g *= random;
	}
	
	match[i] = ((h0 + h1 + h2 + h3) < dSelectBarrier ? 0 : 1);
}

__global__ void gaSelect(int *match, int *dkeepMatching, const int nrVertices, const uint random)
{
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	//Can this vertex still be matched?
	//if (match[i] >= 2) return;
	if (match[i] >= 3) return;

	//Use atomic operations to indicate whether we are done.
	//atomicCAS(&dkeepMatching, 0, 1);
	*dkeepMatching = 1;

	//Start hashing.
	uint h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
	uint a = h0, b = h1, c = h2, d = h3, e, f, g = i;

	for (int j = 0; j < 16; ++j)
	{
		f = (b & c) | ((~b) & d);

		e = d;
		d = c;
		c = b;
		b += LEFTROTATE(a + f + dMD5K[j] + g, dMD5R[j]);
		a = e;

		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;

		g *= random;
	}
	
	match[i] = ((h0 + h1 + h2 + h3) < dSelectBarrier ? 0 : 1);
}


__global__ void gaSelect_from_mis(int *match, int *dkeepMatching, int *L_d,const int nrVertices)
{
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	//Can this vertex still be matched?
	//if (match[i] >= 2) return;
	if (match[i] >= 3) return;

	//Use atomic operations to indicate whether we are done.
	//atomicCAS(&dkeepMatching, 0, 1);
	*dkeepMatching = 1;
	match[i] = L_d[i];
}

__global__ void gMatch(int *match, const int *requests, const int nrVertices)
{

	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	const int r = requests[i];

	//Only unmatched vertices make requests.
	if (r == nrVertices + 1)
	{
		//This is vertex without any available neighbours, discard it.
		match[i] = 3;
		//match[i] = 2;
	}
	else if (r < nrVertices)
	{
		//This vertex has made a valid request.
		if (requests[r] == i)
		{
			//Match the vertices if the request was mutual.
			//match[i] = 4 + min(i, r);
			// I need a pointer to the match for traversal.
			match[i] = 4 + r;

		}
	}
}

//==== Random greedy matching kernels ====
__global__ void grRequest(unsigned int *CP_d,unsigned int *IC_d,int *requests, const int *match, const int nrVertices)
{

	//Let all blue vertices make requests.
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	//const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all blue vertices and let them make requests.
	if (match[i] == 0)
	{
		int dead = 1;
		int k;
		int start = CP_d[i];
		int end = CP_d[i+1];
		//for (int j = indices.x; j < indices.y; ++j)
      	for (k = start;k < end; k++)
		{
			const int ni = IC_d[k];
			const int nm = match[ni];

			//Do we have an unmatched neighbour?
			if (nm < 4 && i!=ni)
			{
				//printf("neighbors of %d : %d\n", i, ni);
				//Is this neighbour red?
				if (nm == 1)
				{
					//Propose to this neighbour.
					requests[i] = ni;
					return;
				}
				
				dead = 0;
			}
		}
		requests[i] = nrVertices + dead;
	}
	else
	{
		//Clear request value.
		requests[i] = nrVertices;
	}

}

__global__ void grRespond(unsigned int *CP_d,unsigned int *IC_d,int *requests, const int *match, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	//const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all red vertices.
	if (match[i] == 1)
	{
		int k;
		int start = CP_d[i];
		int end = CP_d[i+1];
		//Select first available proposer.
		//for (int j = indices.x; j < indices.y; ++j)
      	for (k = start;k < end; k++)
		{
			//const int ni = tex1Dfetch(neighboursTexture, j);
			const int ni = IC_d[k];
			//printf("%d (%d) - %d (%d)\n", i, match[i], ni, match[ni]);
			//Only respond to blue neighbours.
			if (match[ni] == 0)
			{
				//Avoid data thrashing be only looking at the request value of blue neighbours.
				if (requests[ni] == i)
				{
					requests[i] = ni;
					return;
				}
			}
		}
	}
}


//==== Random greedy matching kernels ====
__global__ void grRequestEdgeList(uint64_t *BTypePair_list_d, int *search_tree_src_d, unsigned int *BTypePair_list_counter_d, int *requests, const int *match, const int nrVertices)
{

  // Let all blue vertices make requests.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= BTypePair_list_counter_d[0] || i >= nrVertices)
    return;

  // const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
  uint32_t curr_u = (uint32_t)BTypePair_list_d[i];
  uint32_t curr_v = (BTypePair_list_d[i] >> 32);
  int curr_u_root = search_tree_src_d[curr_u];
  int curr_v_root = search_tree_src_d[curr_v];
  // Look at all blue vertices and let them make requests.
  if (match[curr_u_root] == 0 && match[curr_v_root] == 1)
  {
    requests[curr_u_root] = curr_v;
  }
  else if (match[curr_v_root] == 0 && match[curr_u_root] == 1)
  {
    requests[curr_v_root] = curr_u;
  }
}

__global__ void grRespondEdgeList(uint64_t *BTypePair_list_d, int *search_tree_src_d, unsigned int *BTypePair_list_counter_d, int *requests, const int *match, const int nrVertices)
{
  // Let all blue vertices make requests.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= BTypePair_list_counter_d[0] || i >= nrVertices)
    return;

  uint32_t curr_u = (uint32_t)BTypePair_list_d[i];
  uint32_t curr_v = (BTypePair_list_d[i] >> 32);
  int curr_u_root = search_tree_src_d[curr_u];
  int curr_v_root = search_tree_src_d[curr_v];
  // Look at all blue vertices and let them make requests.
  if (match[curr_u_root] == 0 && match[curr_v_root] == 1 && requests[curr_u_root] == curr_v)
  {
    requests[curr_v_root] = curr_u;
  }
  else if (match[curr_v_root] == 0 && match[curr_u_root] == 1 && requests[curr_v_root] == curr_u)
  {
    requests[curr_u_root] = curr_v;
  }
}

__global__ void gMatchEdgeList(uint64_t *BTypePair_disjoint_list_d, unsigned int *BTypePair_disjoint_list_counter_d, uint64_t *BTypePair_list_d, int *search_tree_src_d, unsigned int *BTypePair_list_counter_d,
                               int *match, const int *requests, const int nrVertices)
{

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= BTypePair_list_counter_d[0] || i >= nrVertices)
    return;

  uint32_t curr_u = (uint32_t)BTypePair_list_d[i];
  uint32_t curr_v = (BTypePair_list_d[i] >> 32);
  int curr_u_root = search_tree_src_d[curr_u];
  int curr_v_root = search_tree_src_d[curr_v];

  const int r_u = requests[curr_u_root];
  const int r_v = requests[curr_v_root];

  if (r_u < nrVertices && r_v < nrVertices)
  {
    // This vertex has made a valid request.
    if (r_u == curr_v && r_v == curr_u && curr_u < curr_v)
    {
      // Match the vertices if the request was mutual.
      // match[i] = 4 + min(i, r);
      //  I need a pointer to the match for traversal.
      match[curr_u_root] = 4 + curr_v;
      match[curr_v_root] = 4 + curr_u;
      uint64_t edgePair = (uint64_t)curr_v << 32 | curr_u;
      int top = atomicAdd(BTypePair_disjoint_list_counter_d, 1);
      BTypePair_disjoint_list_d[top] = edgePair;
    }
  }
}

//==== Weighted greedy matching kernels ====
__global__ void gwRequest(int *requests, const int *match, const int nrVertices)
{
/*
	//Let all blue vertices make requests.
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all blue vertices and let them make requests.
	if (match[i] == 0)
	{
		float maxWeight = -1.0;
		int candidate = nrVertices;
		int dead = 1;

		for (int j = indices.x; j < indices.y; ++j)
		{
			//Only propose to red neighbours.
			const int ni = tex1Dfetch(neighboursTexture, j);
			const int nm = match[ni];

			//Do we have an unmatched neighbour?
			if (nm < 4)
			{
				//Is this neighbour red?
				if (nm == 1)
				{
					//Propose to the heaviest neighbour.
					const float nw = tex1Dfetch(weightsTexture, j);

					if (nw > maxWeight)
					{
						maxWeight = nw;
						candidate = ni;
					}
				}
				
				dead = 0;
			}
		}

		requests[i] = candidate + dead;
	}
	else
	{
		//Clear request value.
		requests[i] = nrVertices;
	}
*/
}

__global__ void gwRespond(int *requests, const int *match, const int nrVertices)
{
/*
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all red vertices.
	if (match[i] == 1)
	{
		float maxWeight = -1;
		int candidate = nrVertices;

		//Select heaviest available proposer.
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = tex1Dfetch(neighboursTexture, j);

			//Only respond to blue neighbours.
			if (match[ni] == 0)
			{
				if (requests[ni] == i)
				{
					const float nw = tex1Dfetch(weightsTexture, j);

					if (nw > maxWeight)
					{
						maxWeight = nw;
						candidate = ni;
					}
				}
			}
		}

		if (candidate < nrVertices)
		{
			requests[i] = candidate;
		}
	}
*/
}


__global__ void extractUnmatched(int *match, int *unmatch, unsigned int *atomicCounter, const int nrVertices)
{

	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	// consider filtering by degree also.
	if (match[i]==-1){
		auto ptr = atomicInc(atomicCounter,1);

	}
}

}
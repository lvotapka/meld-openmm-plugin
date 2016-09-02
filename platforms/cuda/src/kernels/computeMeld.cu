/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#define ELEM_SWAP(a,b) { int t=(a);(a)=(b);(b)=t; }

__device__ float quick_select_float(const float* energy, int *index, int nelems, int select) {
    int low, high, middle, ll, hh;

    low = 0;
    high = nelems - 1;

    for (;;) {
        if (high <= low) { /* One element only */
            return energy[index[select]];
        }

        if (high == low + 1) {  /* Two elements only */
            if (energy[index[low]] > energy[index[high]])
                ELEM_SWAP(index[low], index[high]);
            return energy[index[select]];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (energy[index[middle]] > energy[index[high]])    ELEM_SWAP(index[middle], index[high]);
        if (energy[index[low]]    > energy[index[high]])    ELEM_SWAP(index[low],    index[high]);
        if (energy[index[middle]] > energy[index[low]])     ELEM_SWAP(index[middle], index[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(index[middle], index[low+1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (energy[index[low]] > energy[index[ll]]);
            do hh--; while (energy[index[hh]]  > energy[index[low]]);

            if (hh < ll)
                break;

            ELEM_SWAP(index[ll], index[hh]);
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(index[low], index[hh]);

        /* Re-set active partition */
        if (hh <= select)
            low = ll;
        if (hh >= select)
            high = hh - 1;
    }
}
#undef ELEM_SWAP


__device__ void computeTorsionAngle(const real4* __restrict__ posq, int atom_i, int atom_j, int atom_k, int atom_l,
        float3& r_ij, float3& r_kj, float3& r_kl, float3& m, float3& n,
        float& len_r_kj, float& len_m, float& len_n, float& phi) {
    // compute vectors
    r_ij = trimTo3(posq[atom_j] - posq[atom_i]);
    r_kj = trimTo3(posq[atom_j] - posq[atom_k]);
    r_kl = trimTo3(posq[atom_l] - posq[atom_k]);

    // compute normal vectors
    m = cross(r_ij, r_kj);
    n = cross(r_kj, r_kl);

    // compute lengths
    len_r_kj = sqrt(dot(r_kj, r_kj));
    len_m = sqrt(dot(m, m));
    len_n = sqrt(dot(n, n));

    // compute angle phi
    float x = dot(m / len_m, n / len_n);
    float y = dot(cross(m / len_m, r_kj / len_r_kj), n / len_n);
    phi = atan2(y, x) * 180. / 3.141592654;
}


__device__ void computeTorsionForce(const float dEdPhi, const float3& r_ij, const float3& r_kj, const float3& r_kl,
        const float3& m, const float3& n, const float len_r_kj, const float len_m, const float len_n,
        float3& F_i, float3& F_j, float3& F_k, float3& F_l) {
    F_i = -180. / 3.141592654 * dEdPhi * len_r_kj * m / (len_m * len_m);
    F_l = 180. / 3.141592654 * dEdPhi * len_r_kj * n / (len_n * len_n);
    F_j = -F_i + dot(r_ij, r_kj) / (len_r_kj * len_r_kj) * F_i - dot(r_kl, r_kj) / (len_r_kj * len_r_kj) * F_l;
    F_k = -F_l - dot(r_ij, r_kj) / (len_r_kj * len_r_kj) * F_i + dot(r_kl, r_kj) / (len_r_kj * len_r_kj) * F_l;
}


extern "C" __global__ void computeContacts(const real4* __restrict__ posq,
                                           const int2* __restrict__ atomIndices,
                                           const int numResidues,
                                           int max_threads,
                                           const float contactDist,
                                           int* contacts,
                                           const int* alpha_carbons)
{
//int i_linear_raw = blockIdx.x * blockDim.x + threadIdx.x;
int i_linear;
//int j = blockIdx.y * blockDim.y + threadIdx.y; // cannot use y in openmm it seems...
//int num_iter = ((numResidues * numResidues) / max_threads) + 1;
int counter;

for (int i_linear=blockIdx.x*blockDim.x+threadIdx.x; i_linear<numResidues*numResidues; i_linear+=blockDim.x*gridDim.x) {
        //i_linear = (counter * max_threads) + i_linear_raw; 
        int i = i_linear % numResidues; // integer divide gives the 'column' index, in a way 
        int j = i_linear / numResidues; // modulo gives the 'row' index
        
        real contactDist_sq = contactDist * contactDist;
        //contacts[j + i*numResidues] = 0; // this was a test
        
        if (i < numResidues && j < numResidues) {
                if (i == j) {
                        contacts[j + i*numResidues] = 0;
                } else if ((j == i - 1) || (j == i + 1)) {
                        contacts[j + i*numResidues] = 1;
                // } else if ((j == i - 2) || (j == i + 2)) { // or even +/- 3 ???
                //      contacts[j + i*numResidues] = 0;
                } else {
                        real4 delta = posq[alpha_carbons[i]] - posq[alpha_carbons[j]];
                        real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        //real r = SQRT(distSquared);
                        //if (r < contactDist) {
                        if (distSquared < contactDist_sq) {
                                contacts[j + i*numResidues] = 1;
                        } else {
                                contacts[j + i*numResidues] = 0;
                        }
                }
        }
}
}

extern "C" __global__ void test_get_alpha_carbon_posq(const real4* __restrict__ posq,
                                           float* alphaCarbonPosq,
                                           const int* alpha_carbons,
                                           int numResidues)
{
  // This kernel exists solely for testing purposes to bring over the x,y,z coordinates of the alpha carbons
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if ( index < numResidues ) {
    alphaCarbonPosq[index*3] = (float) posq[alpha_carbons[index]].x; 
    alphaCarbonPosq[index*3 + 1] = (float) posq[alpha_carbons[index]].y; 
    alphaCarbonPosq[index*3 + 2] = (float) posq[alpha_carbons[index]].z; 
  }
}


extern "C" __global__ void dijkstra_initialize(bool* __restrict__ unexplored,
                                           bool* __restrict__ frontier,
                                           int* __restrict__ distance,
                                           int* __restrict__ n_explored,
                                           int src,
                                           int LARGE,
                                           int n_nodes)
{
  // Initialize the arrays right before a Dijkstra calculation
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n_nodes) {
    unexplored[index] = true;
    frontier[index] = false;
    distance[index] = abs(index - src); //LARGE;
    n_explored[index] = 0;
    if (index == src) {
      unexplored[index] = false;
      frontier[index] = true;
      distance[index] = 0;
    }
  }
}

extern "C" __global__ void dijkstra_save_old_vectors(bool* __restrict__ unexplored, // TODO: delete this function: it is deprecated
                                           bool* __restrict__ unexplored_old,
                                           bool* __restrict__ frontier,
                                           bool* __restrict__ frontier_old,
                                           int n_nodes)
{
  // back up the old arrays to be used by the next iteration of the Dijkstra algorithm
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n_nodes) {
    unexplored_old[index] = unexplored[index];
    frontier_old[index] = frontier[index];
  }
}

extern "C" __global__ void dijkstra_settle_and_update(bool* __restrict__ unexplored,
                                           bool* unexplored_old,
                                           bool* frontier,
                                           bool* frontier_old,
                                           int* distance,
                                           int* edge_counts,
                                           int* contacts,
                                           int* n_explored,
                                           int num_iter,
                                           int n_nodes)
{
  // perform a Dijkstra calculation
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int j;
  int edge_index;
  int head;
  if (index < n_nodes) {
    unexplored_old[index] = unexplored[index];
    frontier_old[index] = frontier[index];
  }
  //if (index < n_nodes) {
    n_explored[index] = 0;
    if (unexplored_old[index] == true) { // this node has been unexplored
      for (j=0; j<edge_counts[index]; j++) { // loop thru the nodes that lead to this one 
        edge_index = index*n_nodes + j;  // get the index of this edge
        head = contacts[edge_index];  // the index of the node that is leading to this one
        if (frontier_old[head] == true) { // if the head node is on the frontier, the we need to change our explored status
          frontier[index] = true; // then add myself to the frontier
          unexplored[index] = false; // remove myself from the unexplored
          distance[index] = num_iter + 1; // the number of iterations we've needed to find me is the distance
          n_explored[index]=1;
        }
        
        /*frontier[index] = frontier_old[head];
        unexplored[index] = !frontier_old[head];
        distance[index] = distance[index]*!frontier_old[head] + (num_iter+1)*frontier_old[head];
        n_explored[index]=1;*/
      }
    }
    
    // Is there some way to reduce this to an arithmetic operation?
    
    
  //}

}

extern "C" __global__ void dijkstra_log_reduce(int n_nodes,
                                               int* n_explored,
                                               int* total)
{
  // compute the number of explorations made in this round, and add them to the total
  int i = 1; // the order of this log reduction
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n_nodes) {
    while (i < n_nodes) {
      if ((index % (i*2)) == 0) { // if I belong to this order
        if ((index+i) < n_nodes) {
          n_explored[index] += n_explored[index+i];
        }
      }
      i*=2; // double the order
    }
    if (index == 0) { 
      total[0] = n_explored[0];
    } // set 'total' equal to the reduced sum
  }
}

extern "C" __global__ void assignRestEco(int src,
                                         int numDistRestraints,
                                         int max_threads,
                                         int2* distanceRestResidueIndices,
                                         int* distance,
                                         float* eco_values
                                         )
{
  // If a restraint has a certain "src" value, it will assign the proper ECO value to it

  //int num_iter = (numDistRestraints / max_threads) + 1; // if there are more threads than available, then we can iterate through the higher indeces
  int i, index;
  int dest;
  for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numDistRestraints; index+=blockDim.x*gridDim.x) {
    //if (index < numDistRestraints) {
    dest = distanceRestResidueIndices[index].y;
    if (distanceRestResidueIndices[index].x == src) { // if this restraint has one end in the current src
      eco_values[index] = (float)(1 * distance[dest]); // assign the eco value for this restraint from the distance array
    }
    //}
  }
}

extern "C" __global__ void computeEdgeList(int* contacts,
                                           int* edge_counts,
                                           int num_nodes)
{

int tx = blockIdx.x * blockDim.x + threadIdx.x;

int ptr = 0;
edge_counts[tx] = 0; // this must be initialized at zero

if (tx < num_nodes) {
        for (int i = 0; i < num_nodes; i++) {
                if (contacts[i + tx*num_nodes] == 1) {
                        contacts[ptr + tx*num_nodes] = i;
                        edge_counts[tx]++;
                        ptr++;
                }
        }
}
}


extern "C" __global__ void computeDistRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int2* __restrict__ atomIndices,       // pair of atom indices
                            //const int2* __restrict__ residueIndices,
                            const float4* __restrict__ distanceBounds,  // r1, r2, r3, r4
                            const float* __restrict__ forceConstants,   // k
                            const int* __restrict__ doing_ecos,          // doing_eco
                            const float* __restrict__ eco_factors,         // eco_factor
                            const float* __restrict__ eco_constants,      // eco constants
                            const float* __restrict__ eco_linears,         // eco linear factor
                            float* __restrict__ eco_values,         // the ECO values
                            //float* __restrict__ co_values,          // the CO values (contact order)
                            int* __restrict__ indexToGlobal,            // array of indices into global arrays
                            float* __restrict__ energies,               // global array of restraint energies
                            //float* __restrict__ nonECOenergies,         // global array of non-ECO-modified restraint energies
                            float3* __restrict__ forceBuffer,           // temporary buffer to hold the force
                            const int numRestraints) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) { // clever code to keep updating parameters even if the GPU isn't big enough for them
        // get my global index
        const int globalIndex = indexToGlobal[index];

        // get the distances
        const float r1 = distanceBounds[index].x;
        const float r2 = distanceBounds[index].y;
        const float r3 = distanceBounds[index].z;
        const float r4 = distanceBounds[index].w;

        // get the force constant
        const float k = forceConstants[index];
        const bool doing_eco = doing_ecos[index]; // whether we are doing eco for this restraint
        const float eco_factor = eco_factors[index]; // the factor in the numerator of the eco tweak
        const float eco_constant = eco_constants[index]; 
        const float eco_linear = eco_linears[index];
        float eco_value = eco_values[index]; // the actual ECO value for this restraint
        //const float co_value = co_values[index]; // the CO (contact order) value for this restraint
        //float co_value = 9999; //(float) abs(residueIndices[index].y - residueIndices[index].x);
        //if (eco_value > co_value) { // we don't need to let the ECO be any larger than the CO
        //  eco_value = co_value;
          //eco_values[index] = co_value;
        //}

        // get atom indices and compute distance
        int atomIndexA = atomIndices[index].x;
        int atomIndexB = atomIndices[index].y;
        real4 delta = posq[atomIndexA] - posq[atomIndexB];
        real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real r = SQRT(distSquared);

        // compute force and energy
        float energy = 0.0;
        //float nonECOenergy = 0.0;
        float dEdR = 0.0;
        float diff = 0.0;
        float diff2 = 0.0;
        float force_eco_multiple = 0.0;
        float energy_eco_multiple = 0.0;
        float3 f;

        if(r < r1) {
            energy = k * (r - r1) * (r1 - r2) + 0.5 * k * (r1 - r2) * (r1 - r2);
            dEdR = k * (r1 - r2);
        }
        else if(r < r2) {
            diff = r - r2;
            diff2 = diff * diff;
            energy = 0.5 * k * diff2;
            dEdR = k * diff;
        }
        else if(r < r3) {
            dEdR = 0.0;
            energy = 0.0;
        }
        else if(r < r4) {
            diff = r - r3;
            diff2 = diff * diff;
            energy = 0.5 * k * diff2;
            dEdR = k * diff;
        }
        else {
            energy = k * (r - r4) * (r4 - r3) + 0.5 * k * (r4 - r3) * (r4 - r3);
            dEdR = k * (r4 - r3);
        }
        
        //nonECOenergy = energy;
        
        if ((doing_eco == true) && (eco_value > 0.0)) { // make sure we want to do eco and that the eco value is positive
          assert(eco_value >= 1.0); // This should catch any weird floating point problems
          force_eco_multiple =  (eco_constant + eco_factor / eco_value);
          energy_eco_multiple = (eco_constant + eco_linear*eco_value + eco_factor / eco_value);
          if (force_eco_multiple < 0.0) {
            force_eco_multiple = 0.0; // we don't want a force driving things apart
          }
          if (energy_eco_multiple < 0.0) {
            energy_eco_multiple = 0.0; // we don't want a force driving things apart
          }
          energy *= energy_eco_multiple; // ECO adjustments here
          dEdR   *= force_eco_multiple;
        }

        // store force into local buffer
        if (r > 0) {
            f.x = delta.x * dEdR / r;
            f.y = delta.y * dEdR / r;
            f.z = delta.z * dEdR / r;
        } else {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
        }
        forceBuffer[index] = f;

        // store energy into global buffer
        energies[globalIndex] = energy;
        //nonECOenergies[globalIndex] = nonECOenergy;
    }
}


extern "C" __global__ void computeHyperbolicDistRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int2* __restrict__ atomIndices,       // pair of atom indices
                            const float4* __restrict__ distanceBounds,  // r1, r2, r3, r4
                            const float4* __restrict__ params,          // k1, k2, a, b
                            int* __restrict__ indexToGlobal,            // array of indices into global arrays
                            float* __restrict__ energies,               // global array of restraint energies
                            float3* __restrict__ forceBuffer,           // temporary buffer to hold the force
                            const int numRestraints) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // get my global index
        const int globalIndex = indexToGlobal[index];

        // get the distances
        const float r1 = distanceBounds[index].x;
        const float r2 = distanceBounds[index].y;
        const float r3 = distanceBounds[index].z;
        const float r4 = distanceBounds[index].w;

        // get the parameters
        const float k1 = params[index].x;
        const float k2 = params[index].y;
        const float a = params[index].z;
        const float b = params[index].w;

        // get atom indices and compute distance
        int atomIndexA = atomIndices[index].x;
        int atomIndexB = atomIndices[index].y;
        real4 delta = posq[atomIndexA] - posq[atomIndexB];
        real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real r = SQRT(distSquared);

        // compute force and energy
        float energy = 0.0;
        float dEdR = 0.0;
        float diff = 0.0;
        float diff2 = 0.0;
        float3 f;

        if(r < r1) {
            energy = k1 * (r - r1) * (r1 - r2) + 0.5 * k1 * (r1 - r2) * (r1 - r2);
            dEdR = k1 * (r1 - r2);
        }
        else if(r < r2) {
            diff = r - r2;
            diff2 = diff * diff;
            energy = 0.5 * k1 * diff2;
            dEdR = k1 * diff;
        }
        else if(r < r3) {
            dEdR = 0.0;
            energy = 0.0;
        }
        else if(r < r4) {
            diff = r - r3;
            diff2 = diff * diff;
            energy = 0.5 * k2 * diff2;
            dEdR = k2 * diff;
        }
        else {
            energy = 0.5 * k2 * (b / (r - r3) + a);
            dEdR = -0.5 * b * k2 / (r - r3) / (r - r3);
        }

        // store force into local buffer
        if (r > 0) {
            f.x = delta.x * dEdR / r;
            f.y = delta.y * dEdR / r;
            f.z = delta.z * dEdR / r;
        } else {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
        }
        forceBuffer[index] = f;

        // store energy into global buffer
        energies[globalIndex] = energy;
    }
}


extern "C" __global__ void computeTorsionRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int4* __restrict__ atomIndices,       // indices of atom_{i,j,k,l}
                            const float3* __restrict__ params,          // phi, deltaPhi, forceConstant
                            int* __restrict__ indexToGlobal,            // array of indices into global arrays
                            float* __restrict__ energies,               // global array of restraint energies
                            float3* __restrict__ forceBuffer,           // temporary buffer to hold the force
                                                                        // forceBuffer[index*4] -> atom_i
                                                                        // forceBuffer[index*4 + 3] -> atom_l
                            const int numRestraints) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=gridDim.x*blockDim.x) {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // get the atom indices
        int4 indices = atomIndices[index];
        int atom_i = indices.x;
        int atom_j = indices.y;
        int atom_k = indices.z;
        int atom_l = indices.w;

        // compute the angle and related quantities
        float3 r_ij, r_kj, r_kl;
        float3 m, n;
        float len_r_kj;
        float len_m;
        float len_n;
        float phi;
        computeTorsionAngle(posq, atom_i, atom_j, atom_k, atom_l,
                r_ij, r_kj, r_kl, m, n, len_r_kj, len_m, len_n,  phi);

        // compute E and dE/dphi
        float phiEquil = params[index].x;
        float phiDelta = params[index].y;
        float forceConst = params[index].z;

        float phiDiff = phi - phiEquil;
        if (phiDiff < -180.) {
            phiDiff += 360.;
        } else if (phiDiff > 180.) {
            phiDiff -= 360.;
        }

        float energy = 0.0;
        float dEdPhi = 0.0;
        if (phiDiff < -phiDelta) {
            energy = 0.5 * forceConst * (phiDiff + phiDelta) * (phiDiff + phiDelta);
            dEdPhi = forceConst * (phiDiff + phiDelta);
        }
        else if(phiDiff > phiDelta) {
            energy = 0.5 * forceConst * (phiDiff - phiDelta) * (phiDiff - phiDelta);
            dEdPhi = forceConst * (phiDiff - phiDelta);
        }
        else{
            energy = 0.0;
            dEdPhi = 0.0;
        }

        energies[globalIndex] = energy;

        computeTorsionForce(dEdPhi, r_ij, r_kj, r_kl, m, n, len_r_kj, len_m, len_n,
                forceBuffer[4 * index + 0], forceBuffer[4 * index + 1],
                forceBuffer[4 * index + 2], forceBuffer[4 * index + 3]);
    }
}


extern "C" __global__ void computeDistProfileRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int2* __restrict__ atomIndices,       // pair of atom indices
                            const float2* __restrict__ distRanges,      // upper and lower bounds of spline
                            const int* __restrict__ nBins,              // number of bins
                            const float4* __restrict__ splineParams,    // a0, a1, a2, a3
                            const int2* __restrict__ paramBounds,       // upper and lower bounds for each spline
                            const float* __restrict__ scaleFactor,      // scale factor for energies and forces
                            const int* __restrict__ indexToGlobal,      // index of this restraint in the global array
                            float* __restrict__ restraintEnergies,      // global energy of each restraint
                            float3* __restrict__ restraintForce,        // cache the forces for application later
                            const int numRestraints ) {

    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // get atom indices and compute distance
        int atomIndexA = atomIndices[index].x;
        int atomIndexB = atomIndices[index].y;

        real4 delta = posq[atomIndexA] - posq[atomIndexB];
        real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real r = SQRT(distSquared);

        // compute bin
        int bin = (int)( floor((r - distRanges[index].x) / (distRanges[index].y - distRanges[index].x) * nBins[index]) );

        // compute the force and energy
        float energy = 0.0;
        float dEdR = 0.0;
        float binWidth = (distRanges[index].y - distRanges[index].x) / nBins[index];
        if (bin < 0){
            energy = scaleFactor[index] * splineParams[paramBounds[index].x].x;
        }
        else if (bin >= nBins[index]) {
            energy = scaleFactor[index] * (splineParams[paramBounds[index].y - 1].x +
                                           splineParams[paramBounds[index].y - 1].y +
                                           splineParams[paramBounds[index].y - 1].z +
                                           splineParams[paramBounds[index].y - 1].w);
        }
        else {
            float t = (r - bin * binWidth + distRanges[index].x) / binWidth;
            float a0 = splineParams[ paramBounds[index].x + bin ].x;
            float a1 = splineParams[ paramBounds[index].x + bin ].y;
            float a2 = splineParams[ paramBounds[index].x + bin ].z;
            float a3 = splineParams[ paramBounds[index].x + bin ].w;
            energy = scaleFactor[index] * (a0 + a1 * t + a2 * t * t + a3 * t * t * t);
            dEdR = scaleFactor[index] * (a1 + 2.0 * a2 * t + 3.0 * a3 * t * t) / binWidth;
        }

        // store force into local buffer
        float3 f;
        f.x = delta.x * dEdR / r;
        f.y = delta.y * dEdR / r;
        f.z = delta.z * dEdR / r;
        restraintForce[index] = f;

        // store energy into global buffer
        restraintEnergies[globalIndex] = energy;
    }
}

extern "C" __global__ void computeTorsProfileRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int4* __restrict__ atomIndices0,      // i,j,k,l for torsion 0
                            const int4* __restrict__ atomIndices1,      // i,j,k,l for torsion 1
                            const int* __restrict__ nBins,              // number of bins
                            const float4* __restrict__ params0,         // a0 - a3
                            const float4* __restrict__ params1,         // a4 - a7
                            const float4* __restrict__ params2,         // a8 - a11
                            const float4* __restrict__ params3,         // a12 - a15
                            const int2* __restrict__ paramBounds,       // upper and lower bounds for each spline
                            const float* __restrict__ scaleFactor,      // scale factor for energies and forces
                            const int* __restrict__ indexToGlobal,      // index of this restraint in the global array
                            float* __restrict__ restraintEnergies,      // global energy of each restraint
                            float3* __restrict__ forceBuffer,        // cache the forces for application later
                            const int numRestraints ) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=gridDim.x*blockDim.x) {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // compute phi
        int phi_atom_i = atomIndices0[index].x;
        int phi_atom_j = atomIndices0[index].y;
        int phi_atom_k = atomIndices0[index].z;
        int phi_atom_l = atomIndices0[index].w;
        float3 phi_r_ij, phi_r_kj, phi_r_kl;
        float3 phi_m, phi_n;
        float phi_len_r_kj;
        float phi_len_m;
        float phi_len_n;
        float phi;
        computeTorsionAngle(posq, phi_atom_i, phi_atom_j, phi_atom_k, phi_atom_l,
                phi_r_ij, phi_r_kj, phi_r_kl, phi_m, phi_n, phi_len_r_kj, phi_len_m, phi_len_n, phi);

        // compute psi
        int psi_atom_i = atomIndices1[index].x;
        int psi_atom_j = atomIndices1[index].y;
        int psi_atom_k = atomIndices1[index].z;
        int psi_atom_l = atomIndices1[index].w;
        float3 psi_r_ij, psi_r_kj, psi_r_kl;
        float3 psi_m, psi_n;
        float psi_len_r_kj;
        float psi_len_m;
        float psi_len_n;
        float psi;
        computeTorsionAngle(posq, psi_atom_i, psi_atom_j, psi_atom_k, psi_atom_l,
                psi_r_ij, psi_r_kj, psi_r_kl, psi_m, psi_n, psi_len_r_kj, psi_len_m, psi_len_n, psi);

        // compute bin indices
        int i = (int)(floor((phi + 180.)/360. * nBins[index]));
        int j = (int)(floor((psi + 180.)/360. * nBins[index]));

        if (i >= nBins[index]) {
            i = 0;
            phi -= 360.;
        }
        if (i < 0) {
            i = nBins[index] - 1;
            phi += 360.;
        }

        if (j >= nBins[index]) {
            j = 0;
            psi -= 360.;
        }
        if (j < 0) {
            j = nBins[index] - 1;
            psi += 360.;
        }

        float delta = 360. / nBins[index];
        float u = (phi - i * delta + 180.) / delta;
        float v = (psi - j * delta + 180.) / delta;

        int pi = paramBounds[index].x + i * nBins[index] + j;

        float energy = params0[pi].x         + params0[pi].y * v       + params0[pi].z * v*v       + params0[pi].w * v*v*v +
                       params1[pi].x * u     + params1[pi].y * u*v     + params1[pi].z * u*v*v     + params1[pi].w * u*v*v*v +
                       params2[pi].x * u*u   + params2[pi].y * u*u*v   + params2[pi].z * u*u*v*v   + params2[pi].w * u*u*v*v*v +
                       params3[pi].x * u*u*u + params3[pi].y * u*u*u*v + params3[pi].z * u*u*u*v*v + params3[pi].w * u*u*u*v*v*v;
        energy = energy * scaleFactor[index];

        float dEdPhi = params1[pi].x         + params1[pi].y * v     + params1[pi].z * v*v     + params1[pi].w * v*v*v +
                       params2[pi].x * 2*u   + params2[pi].y * 2*u*v   + params2[pi].z * 2*u*v*v   + params2[pi].w * 2*u*v*v*v +
                       params3[pi].x * 3*u*u + params3[pi].y * 3*u*u*v + params3[pi].z * 3*u*u*v*v + params3[pi].w * 3*u*u*v*v*v;
        dEdPhi = dEdPhi * scaleFactor[index] / delta;

        float dEdPsi = params0[pi].y         + params0[pi].z * 2*v       + params0[pi].w * 3*v*v +
                       params1[pi].y * u     + params1[pi].z * u*2*v     + params1[pi].w * u*3*v*v +
                       params2[pi].y * u*u   + params2[pi].z * u*u*2*v   + params2[pi].w * u*u*3*v*v +
                       params3[pi].y * u*u*u + params3[pi].z * u*u*u*2*v + params3[pi].w * u*u*u*3*v*v;
        dEdPsi = dEdPsi * scaleFactor[index] / delta;

        restraintEnergies[globalIndex] = energy;

        computeTorsionForce(dEdPhi, phi_r_ij, phi_r_kj, phi_r_kl, phi_m, phi_n, phi_len_r_kj, phi_len_m, phi_len_n,
                forceBuffer[8 * index + 0], forceBuffer[8 * index + 1],
                forceBuffer[8 * index + 2], forceBuffer[8 * index + 3]);
        computeTorsionForce(dEdPsi, psi_r_ij, psi_r_kj, psi_r_kl, psi_m, psi_n, psi_len_r_kj, psi_len_m, psi_len_n,
                forceBuffer[8 * index + 4], forceBuffer[8 * index + 5],
                forceBuffer[8 * index + 6], forceBuffer[8 * index + 7]);
    }
}

extern "C" __global__ void computeCartProfileRest(
                                        const real4* __restrict__ posq, const int* atom_indices,
                                        const float* coeffs, const int* starting_coeffs,
                                        const float3* dims, const float3* resolution,
                                        const float3* origin, const float* scale_factor,
                                        const int* global_indices, float* __restrict__ energies,
                                        float3* __restrict__ force_buffer, float3* pos_buffer, const int numRestraints) {

for (int tx=blockIdx.x*blockDim.x+threadIdx.x; tx<numRestraints; tx+=blockDim.x*gridDim.x) {
            
            float x = floor((posq[atom_indices[tx]].x - origin[tx].x)/resolution[tx].x);
            float y = floor((posq[atom_indices[tx]].y - origin[tx].y)/resolution[tx].y);
            float z = floor((posq[atom_indices[tx]].z - origin[tx].z)/resolution[tx].z);
            
            float dx = posq[atom_indices[tx]].x - (origin[tx].x + x*resolution[tx].x);
            float dy = posq[atom_indices[tx]].y - (origin[tx].y + y*resolution[tx].y); // the position of the atom in relation to the corner of this bin
            float dz = posq[atom_indices[tx]].z - (origin[tx].z + z*resolution[tx].z);
            
            bool out_of_bounds = false;
            
            if (x < 0) { // make sure that we fall within the boundaries of the grid
              x = 0;
              dx = 0;
              out_of_bounds = true;
            } 
            if (y < 0) {
              y = 0;
              dy = 0;
              out_of_bounds = true;
            } 
            if (z < 0) {
              z = 0;
              dz = 0;
              out_of_bounds = true;
            }
            if (x > dims[tx].x) {
              x = dims[tx].x;
              dx = resolution[tx].x;
              out_of_bounds = true;
            } 
            if (y > dims[tx].y) {
              y = dims[tx].y;
              dy = resolution[tx].y;
              out_of_bounds = true;
            } 
            if (z > dims[tx].z) {
              z = dims[tx].z;
              dz = resolution[tx].z;
              out_of_bounds = true;
            } 
            
            

            //int count_x = dims[tx].x/resolution[tx].x;
            //int count_y = dims[tx].y/resolution[tx].y; // should not be necessary
            //int count_z = dims[tx].z/resolution[tx].z;

            //int bin = x*count_x*count_z + y*count_z + z; // not necessary
            int bin = x*dims[tx].y*dims[tx].z + y*dims[tx].z + z;

            float energy = 0;
            float3 force;
            force.x = 0; 
            force.y = 0; 
            force.z = 0;
            float norm_dx = dx / resolution[tx].x;
            float norm_dy = dy / resolution[tx].y; // Does not need to be computed every time
            float norm_dz = dz / resolution[tx].z;
            
            // DEBUG
            pos_buffer[tx].x = norm_dx;
            pos_buffer[tx].y = norm_dy;
            pos_buffer[tx].z = norm_dz;
            
            assert(isfinite(norm_dx));
            assert(isfinite(norm_dy));
            assert(isfinite(norm_dz));
            
            for (int pow_x = 0; pow_x < 4; pow_x++) {
                for (int pow_y = 0; pow_y < 4; pow_y++) {
                    for (int pow_z = 0; pow_z < 4; pow_z++) {
                        int a_index = pow_x + 4*pow_y + 16*pow_z;
                        
                        float coefficient = coeffs[starting_coeffs[tx] + 64*bin + a_index];
                        assert(isfinite(coefficient));
                        energy = energy + coefficient*pow(norm_dx, (float) pow_x)*pow(norm_dy, (float) pow_y)*pow(norm_dz, (float) pow_z);
                        if (pow_x > 0) {
                            force.x = force.x - coefficient *
                                               pow_x*pow(norm_dx, (float) (pow_x-1) )*pow(norm_dy, (float) pow_y)*pow(norm_dz, (float) pow_z); }
                        if (pow_y > 0) {
                            force.y = force.y - coefficient *
                                               pow(norm_dx, (float) pow_x)*pow_y*pow(norm_dy, (float) (pow_y-1) )*pow(norm_dz, (float) pow_z); } // ALERT: this might be a problem for detailed balance if out of grid boundary
                        if (pow_z > 0) {
                            force.z = force.z - coefficient *
                                               pow(norm_dx, (float) pow_x)*pow(norm_dy, (float) pow_y)*pow_z*pow(norm_dz, (float) (pow_z-1) ); }

                    }
                }
            }
            
            /*
            if (out_of_bounds == true) { // If we go out of bounds, then just set the force to zero.
                force.x = 0; 
                force.y = 0; 
                force.z = 0;
            }
            */
            
            assert(isfinite(energy));
            assert(isfinite(force.x));
            assert(isfinite(force.y));
            assert(isfinite(force.z));
                 
            energies[global_indices[tx]] = energy * scale_factor[tx];
            force_buffer[tx].x = force.x * scale_factor[tx];
            force_buffer[tx].y = force.y * scale_factor[tx];
            force_buffer[tx].z = force.z * scale_factor[tx];
}
}



extern "C" __global__ void evaluateAndActivate(
        const int numGroups,
        const int* __restrict__ numActiveArray,
        const int2* __restrict__ boundsArray,
        const int* __restrict__ pristineIndexArray,
        int* __restrict__ tempIndexArray,
        const float* __restrict__ energyArray,
        float* __restrict__ activeArray,
        float* __restrict__ targetEnergyArray)
{
    // This kernel computes which restraints are active within each group.
    // It uses "warp-level" programming to do this, where each warp within
    // a threadblock computes the results for a single group. All threads
    // within each group are implicity synchronized at the hardware
    // level.

    // These are runtime parameters set tby the C++ code.
    const int groupsPerBlock = GROUPSPERBLOCK;
    const int maxGroupSize = MAXGROUPSIZE;

    // Because each warp is computing a separate interaction, we need to
    // keep track of which block we are acting on and our index within
    // that warp.
    const int groupOffsetInBlock = threadIdx.x / 32;
    const int threadOffsetInWarp = threadIdx.x % 32;

    // We store the energies and indices into scratch buffers. These scratch
    // buffers are also used for reductions within each warp.
    extern __shared__ volatile char scratch[];
    volatile float* warpScratchEnergy = (float*)&scratch[groupOffsetInBlock*maxGroupSize*(sizeof(float)+sizeof(int))];
    volatile int* warpScratchIndices = (int*)&scratch[groupOffsetInBlock*maxGroupSize*(sizeof(float)+sizeof(int)) +
                                                      maxGroupSize*sizeof(float)];
    volatile float* warpReductionBuffer = (float*)&scratch[groupOffsetInBlock*32*sizeof(float)];

    // each warp loads the energies and indices for a group
    for (int groupIndex=groupsPerBlock*blockIdx.x+groupOffsetInBlock; groupIndex<numGroups; groupIndex+=groupsPerBlock*gridDim.x) {
        const int numActive = numActiveArray[groupIndex];
        const int start = boundsArray[groupIndex].x;
        const int end = boundsArray[groupIndex].y;
        const int length = end - start;
        const bool applyAll = (numActive == length);

        // copy the energies to shared memory and setup indices
        if (!applyAll) {
            for(int i=threadOffsetInWarp; i<length; i+=32) {
                const float energy = energyArray[pristineIndexArray[i + start]];
                warpScratchIndices[i] = i;
                warpScratchEnergy[i] = energy;
            }
        }

        // now, we run the quick select algorithm.
        // this is not parallelized, so we only run it on one thread
        // per block.
        if (threadOffsetInWarp==0) {
            float energyCut = 0.0;
            if (!applyAll) {
                energyCut = quick_select_float((const float*)warpScratchEnergy, (int *)warpScratchIndices, length, numActive-1);
            }
            else {
                energyCut = 9.99e99;
            }
            warpScratchEnergy[0] = energyCut;
        }


        // now we're back on all threads again
        float energyCut = warpScratchEnergy[0];
        float thisActive = 0.0;
        float thisEnergy = 0.0;

        // we are going to start writing to warpReductionBuffer,
        // which may overlap with the warpScratch* buffers, so
        // we need to make sure that all threads are done first.
        __syncthreads();

        // reset the reduction buffers to zero
        warpReductionBuffer[threadOffsetInWarp] = 0.0;

        // sum up the energy for each restraint
        for(int i=threadOffsetInWarp+start; i<end; i+=32) {
            thisEnergy = energyArray[pristineIndexArray[i]];
            thisActive = (float)(thisEnergy <= energyCut);
            activeArray[pristineIndexArray[i]] = thisActive;
            warpReductionBuffer[threadOffsetInWarp] += thisActive * thisEnergy;
        }

        // now we do a parallel reduction within each warp
        int totalThreads = 32;
        int index2 = 0;
        while (totalThreads > 1) {
            int halfPoint = (totalThreads >> 1);
            if (threadOffsetInWarp < halfPoint) {
                index2 = threadOffsetInWarp + halfPoint;
                warpReductionBuffer[threadOffsetInWarp] += warpReductionBuffer[index2];
            }
            totalThreads = halfPoint;
        }

        // now store the energy for this group
        if (threadOffsetInWarp == 0) {
            targetEnergyArray[groupIndex] = warpReductionBuffer[0];
        }

        // make sure we're all done before we start again
        __syncthreads();
    }
}


__device__ void findMinMax(int length, const float* energyArray, float* minBuffer, float* maxBuffer) {
    const int tid = threadIdx.x;
    float energy;
    float min = 9.9e99;
    float max = -9.9e99;
    // Each thread computes the min and max for it's energies and stores them in the buffers
    for (int i=tid; i<length; i+=blockDim.x) {
        energy = energyArray[i];
        if (energy < min) {
            min = energy;
        }
        if (energy > max) {
            max = energy;
        }
    }
    minBuffer[tid] = min;
    maxBuffer[tid] = max;
    __syncthreads();

    // Now we do a parallel reduction
    int totalThreads = blockDim.x;
    int index2 = 0;
    float temp = 0;
    while (totalThreads > 1) {
        int halfPoint = (totalThreads >> 1);
        if (tid < halfPoint) {
            index2 = tid + halfPoint;
            temp = minBuffer[index2];
            if (temp < minBuffer[tid]) {
                minBuffer[tid] = temp;
            }
            temp = maxBuffer[index2];
            if (temp > maxBuffer[tid]) {
                maxBuffer[tid] = temp;
            }
        }
        __syncthreads();
        totalThreads = halfPoint;
    }
    __syncthreads();
}


extern "C" __global__ void evaluateAndActivateCollections(
        const int numCollections,
        const int* __restrict__ numActiveArray,
        const int2* __restrict__ boundsArray,
        const int* __restrict__ indexArray,
        const float* __restrict__ energyArray,
        float* __restrict__ activeArray,
        int * __restrict__ encounteredNaN)
{
    const float TOLERANCE = 1e-4;
    const int maxCollectionSize = MAXCOLLECTIONSIZE;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;  // which thread are we within this warp

    // shared memory:
    // energyBuffer: maxCollectionSize floats
    // min/max Buffer: gridDim.x floats
    // binCounts: gridDim.x ints
    extern __shared__ char collectionScratch[];
    float* energyBuffer = (float*)&collectionScratch[0];
    float* minBuffer = (float*)&collectionScratch[maxCollectionSize*sizeof(float)];
    float* maxBuffer = (float*)&collectionScratch[(maxCollectionSize+blockDim.x)*sizeof(float)];
    int* binCounts = (int*)&collectionScratch[(maxCollectionSize+2*blockDim.x)*sizeof(float)];
    int* bestBin = (int*)&(collectionScratch[(maxCollectionSize + 2 * blockDim.x) * sizeof(float) +
                                             blockDim.x * sizeof(int)]);

    for (int collIndex=blockIdx.x; collIndex<numCollections; collIndex+=gridDim.x) {
        // we need to find the value of the cutoff energy below, then we will
        // activate all groups with lower energy
        float energyCutoff = 0.0;

        int numActive = numActiveArray[collIndex];
        int start = boundsArray[collIndex].x;
        int end = boundsArray[collIndex].y;
        int length = end - start;

        // load the energy buffer for this collection
        for (int i=tid; i<length; i+=blockDim.x) {
            const float energy = energyArray[indexArray[start + i]];
            energyBuffer[i] = energy;
        }
        __syncthreads();

        findMinMax(length, energyBuffer, minBuffer, maxBuffer);
        float min = minBuffer[0];
        float max = maxBuffer[0];
        float delta = max - min;


        // All of the energies are the same, so they should all be active.
        // Note: we need to break out here in this case, as otherwise delta
        // will be zero and bad things will happen
        if (fabs(max-min) < TOLERANCE) {
            energyCutoff = max;
        } else {
            // Here we need to find the k'th highest energy. We do this using a recursive,
            // binning and counting strategy. We divide the interval (min, max) into blockDim.x
            // bins. We assign each energy to a bin, increment the count, and update
            // the min and max. Then, we find the bin that contains the k'th lowest energy. If
            // min==max for this bin, then we are done. Otherwise, we set the new (min, max) for
            // the bins and recompute, assigning energies less than min to bin 0.

            // loop until we break out at convergence
            for (;;) {

                // check to see if have encountered NaN, which will
                // result in an infinite loop
                if(tid==0) {
                    if (!isfinite(min) || !isfinite(max)) {
                        *encounteredNaN = 1;
                    }
                }
                // zero out the buffers
                binCounts[tid] = 0;
                minBuffer[tid] = 9.0e99;
                maxBuffer[tid] = 0.0;
                __syncthreads();

                // If we hit a NaN then abort early now that encounteredNaN is set.
                // This will cause an exception on the C++ side
                if (*encounteredNaN) {
                    return;
                }

                // loop over all energies
                for (int i=tid; i<length; i+=blockDim.x) {
                    float energy = energyBuffer[i];
                    // compute which bin this energy lies in
                    int index = float2int(floorf((blockDim.x-1) / delta * (energy - min)));

                    // we only count entries that lie within min and max
                    if ( (index >= 0) && (index < blockDim.x) ) {

                        // increment the counter using atomic function
                        atomicAdd(&binCounts[index], 1);
                        // update the min and max bounds for the bin using atomic functions
                        // note we need to cast to an integer, but floating point values
                        // still compare correctly when represented as integers
                        // this assumes that all energies are >0
                        atomicMin((unsigned int*)&minBuffer[index], __float_as_int(energy));
                        atomicMax((unsigned int*)&maxBuffer[index], __float_as_int(energy));
                    }
                }
                // make sure all threads are done
                __syncthreads();

                // now we need to do a cumulative sum, also known as an inclusive scan
                // we will do this using a fast three-phase parallel algorithm
                // this code assumes 1024 threads in 32 warps of 32 threads
                // it will require modification to work with arbitrary sizes

                // first, we do the cumulative sum within each warp
                // this works because the threads are all implicity synchronized
                if (lane >= 1) binCounts[tid] += binCounts[tid - 1];
                if (lane >= 2) binCounts[tid] += binCounts[tid - 2];
                if (lane >= 4) binCounts[tid] += binCounts[tid - 4];
                if (lane >= 8) binCounts[tid] += binCounts[tid - 8];
                if (lane >= 16) binCounts[tid] += binCounts[tid - 16];
                __syncthreads();

                // now we use a single thread to do a cumulative sum over the last elements of each
                // of the 32 warps
                if (warp == 0) {
                    if (lane >= 1) binCounts[32 * tid + 31] += binCounts[32 * (tid - 1) + 31];
                    if (lane >= 2) binCounts[32 * tid + 31] += binCounts[32 * (tid - 2) + 31];
                    if (lane >= 4) binCounts[32 * tid + 31] += binCounts[32 * (tid - 4) + 31];
                    if (lane >= 8) binCounts[32 * tid + 31] += binCounts[32 * (tid - 8) + 31];
                    if (lane >= 16) binCounts[32 * tid + 31] += binCounts[32 * (tid - 16) + 31];
                }
                __syncthreads();

                // new each warp adds the value of the 31st element of the previous warp
                // there is nothing to add for warp0, so we skip it
                // the last element of each warp already has this sum from the previous step,
                // so we skip it
                if (warp>0 && lane<31) {
                    binCounts[tid] += binCounts[32 * warp - 1];
                }
                __syncthreads();

                // now we need to find the bin containing the k'th highest value
                // we use a single warp, where each thread looks at a block of 32 entries
                // to find the smallest index where the cumulative sum is >= numActive
                // we set flag if we find one
                // this section uses implicit synchronization between threads in a single warp
                if (warp == 0) {
                    int counter = 0;
                    int flag = false;
                    for (counter=0; counter<32; counter++) {
                        if (binCounts[32 * tid + counter] >= numActive) {
                            flag = true;
                            break;
                        }
                    }
                    // now find the smallest bin that meets the criteria
                    if (tid == 0) {
                        *bestBin = 1025;
                    }
                    // if we found a value >= numActive, then update the minimum value
                    if (flag) {
                        atomicMin(bestBin, 32 * tid + counter);
                    }
                }
                __syncthreads();

                const float binMin = minBuffer[*bestBin];
                const float binMax = maxBuffer[*bestBin];

                //  if all energies in this bin are the same, then we are done
                if (fabs(binMin-binMax) < TOLERANCE) {
                    energyCutoff = binMax;
                    break;
                }

                // if this bin ends exactly on the k'th lowest energy, then we are done
                if (binCounts[*bestBin] == numActive) {
                    energyCutoff = binMax;
                    break;
                }

                // otherwise, the correct value lies somewhere within this bin
                // it will between binMin and binMax and we need to find the
                // binCounts[*bestBin] - numActive 'th element
                // we loop through again searching with these updated parameters
                min = binMin;
                max = binMax;
                delta = max - min;
                numActive = binCounts[*bestBin] - numActive;
                __syncthreads();
            }
        }

        // now we know the energyCutoff, so apply it to each group
        for (int i=tid; i<length; i+=blockDim.x) {
            if (energyBuffer[i] <= energyCutoff) {
                activeArray[indexArray[i + start]] = 1.0;
            }
            else {
                activeArray[indexArray[i + start]] = 0.0;
            }
        }
        __syncthreads();
    }
}


extern "C" __global__ void applyGroups(
                            float* __restrict__ groupActive,
                            float* __restrict__ restraintActive,
                            const int2* __restrict__ bounds,
                            int numGroups) {
    for (int groupIndex=blockIdx.x; groupIndex<numGroups; groupIndex+=gridDim.x) {
        float active = groupActive[groupIndex];
        for (int i=bounds[groupIndex].x + threadIdx.x; i<bounds[groupIndex].y; i+=blockDim.x) {
            restraintActive[i] *= active;
        }
    }
}


extern "C" __global__ void applyDistRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
                                const int2* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                //const float* __restrict__ globalNonEcoEnergies,
                                const float* __restrict__ globalActive,
                                const int numDistRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numDistRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex].x;
            int index2 = atomIndices[restraintIndex].y;
            // comment out both lines below to completely disregard any MELD energy contributions to the Replica exchange probabilities
            energyAccum += globalEnergies[globalIndex]; // the old way, using energies with ECO and all
            //energyAccum += globalNonEcoEnergies[globalIndex]; // the new way, using energies without ECO
            float3 f = restForces[restraintIndex];

            atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
            atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
            atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

            atomicAdd(&force[index2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
            atomicAdd(&force[index2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
            atomicAdd(&force[index2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyHyperbolicDistRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
                                const int2* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numDistRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numDistRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex].x;
            int index2 = atomIndices[restraintIndex].y;
            energyAccum += globalEnergies[globalIndex];
            float3 f = restForces[restraintIndex];

            atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
            atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
            atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

            atomicAdd(&force[index2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
            atomicAdd(&force[index2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
            atomicAdd(&force[index2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyTorsionRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
                                const int4* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int atom_i = atomIndices[restraintIndex].x;
            int atom_j = atomIndices[restraintIndex].y;
            int atom_k = atomIndices[restraintIndex].z;
            int atom_l = atomIndices[restraintIndex].w;
            energyAccum += globalEnergies[globalIndex];

            // update forces
            float3 f_i = restForces[restraintIndex * 4 + 0];
            float3 f_j = restForces[restraintIndex * 4 + 1];
            float3 f_k = restForces[restraintIndex * 4 + 2];
            float3 f_l = restForces[restraintIndex * 4 + 3];

            atomicAdd(&force[atom_i],                        static_cast<unsigned long long>((long long) (f_i.x*0x100000000)));
            atomicAdd(&force[atom_i  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_i.y*0x100000000)));
            atomicAdd(&force[atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_i.z*0x100000000)));

            atomicAdd(&force[atom_j],                        static_cast<unsigned long long>((long long) (f_j.x*0x100000000)));
            atomicAdd(&force[atom_j  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_j.y*0x100000000)));
            atomicAdd(&force[atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_j.z*0x100000000)));

            atomicAdd(&force[atom_k],                        static_cast<unsigned long long>((long long) (f_k.x*0x100000000)));
            atomicAdd(&force[atom_k  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_k.y*0x100000000)));
            atomicAdd(&force[atom_k + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_k.z*0x100000000)));

            atomicAdd(&force[atom_l],                        static_cast<unsigned long long>((long long) (f_l.x*0x100000000)));
            atomicAdd(&force[atom_l  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_l.y*0x100000000)));
            atomicAdd(&force[atom_l + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_l.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyDistProfileRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
                                const int2* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex].x;
            int index2 = atomIndices[restraintIndex].y;
            energyAccum += globalEnergies[globalIndex];
            float3 f = restForces[restraintIndex];

            atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
            atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
            atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

            atomicAdd(&force[index2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
            atomicAdd(&force[index2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
            atomicAdd(&force[index2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyTorsProfileRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
                                const int4* __restrict__ atomIndices0,
                                const int4* __restrict__ atomIndices1,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            // update energy
            energyAccum += globalEnergies[globalIndex];

            // update phi
            int phi_atom_i = atomIndices0[restraintIndex].x;
            int phi_atom_j = atomIndices0[restraintIndex].y;
            int phi_atom_k = atomIndices0[restraintIndex].z;
            int phi_atom_l = atomIndices0[restraintIndex].w;

            // update forces
            float3 phi_f_i = restForces[restraintIndex * 8 + 0];
            float3 phi_f_j = restForces[restraintIndex * 8 + 1];
            float3 phi_f_k = restForces[restraintIndex * 8 + 2];
            float3 phi_f_l = restForces[restraintIndex * 8 + 3];

            atomicAdd(&force[phi_atom_i],                        static_cast<unsigned long long>((long long) (phi_f_i.x*0x100000000)));
            atomicAdd(&force[phi_atom_i + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_i.y*0x100000000)));
            atomicAdd(&force[phi_atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_i.z*0x100000000)));

            atomicAdd(&force[phi_atom_j],                        static_cast<unsigned long long>((long long) (phi_f_j.x*0x100000000)));
            atomicAdd(&force[phi_atom_j + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_j.y*0x100000000)));
            atomicAdd(&force[phi_atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_j.z*0x100000000)));

            atomicAdd(&force[phi_atom_k],                        static_cast<unsigned long long>((long long) (phi_f_k.x*0x100000000)));
            atomicAdd(&force[phi_atom_k + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_k.y*0x100000000)));
            atomicAdd(&force[phi_atom_k + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_k.z*0x100000000)));

            atomicAdd(&force[phi_atom_l],                        static_cast<unsigned long long>((long long) (phi_f_l.x*0x100000000)));
            atomicAdd(&force[phi_atom_l + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_l.y*0x100000000)));
            atomicAdd(&force[phi_atom_l + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_l.z*0x100000000)));

            // update psi
            int psi_atom_i = atomIndices1[restraintIndex].x;
            int psi_atom_j = atomIndices1[restraintIndex].y;
            int psi_atom_k = atomIndices1[restraintIndex].z;
            int psi_atom_l = atomIndices1[restraintIndex].w;

            // update forces
            float3 psi_f_i = restForces[restraintIndex * 8 + 4];
            float3 psi_f_j = restForces[restraintIndex * 8 + 5];
            float3 psi_f_k = restForces[restraintIndex * 8 + 6];
            float3 psi_f_l = restForces[restraintIndex * 8 + 7];

            atomicAdd(&force[psi_atom_i],                        static_cast<unsigned long long>((long long) (psi_f_i.x*0x100000000)));
            atomicAdd(&force[psi_atom_i + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_i.y*0x100000000)));
            atomicAdd(&force[psi_atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_i.z*0x100000000)));

            atomicAdd(&force[psi_atom_j],                        static_cast<unsigned long long>((long long) (psi_f_j.x*0x100000000)));
            atomicAdd(&force[psi_atom_j + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_j.y*0x100000000)));
            atomicAdd(&force[psi_atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_j.z*0x100000000)));

            atomicAdd(&force[psi_atom_k],                        static_cast<unsigned long long>((long long) (psi_f_k.x*0x100000000)));
            atomicAdd(&force[psi_atom_k + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_k.y*0x100000000)));
            atomicAdd(&force[psi_atom_k + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_k.z*0x100000000)));

            atomicAdd(&force[psi_atom_l],                        static_cast<unsigned long long>((long long) (psi_f_l.x*0x100000000)));
            atomicAdd(&force[psi_atom_l + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_l.y*0x100000000)));
            atomicAdd(&force[psi_atom_l + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_l.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}

extern "C" __global__ void applyCartProfileRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
                                const int* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
                                
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex];
            energyAccum += globalEnergies[globalIndex];
            float3 f = restForces[restraintIndex];

            atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
            atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
            atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


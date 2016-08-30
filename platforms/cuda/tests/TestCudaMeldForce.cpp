/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#include "MeldForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerMeldCudaKernelFactories();


void testDistRest() {
    // setup system
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    system.addParticle(1.0);

    // setup meld force
    MeldForce* force = new MeldForce();
    int k = 1.0;
    bool doing_eco = false;
    float eco_factor = 1.0;
    float eco_constant = 0.0;
    float eco_linear = 0.0;
    int res_index1 = 0;
    int res_index2 = 0;
    int restIdx = force->addDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k, doing_eco, eco_factor, eco_constant, eco_linear, res_index1, res_index2);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // setup the context
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // There are five regions:
    // I:       r < 1
    // II:  1 < r < 2
    // III: 2 < r < 3
    // IV:  3 < r < 4
    // V:   4 < r

    // test region I
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.5, 0.0, 0.0);
    context.setPositions(positions);

    float expectedEnergy = 1.0;
    Vec3 expectedForce = Vec3(-1.0, 0.0, 0.0);

    State stateI = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateI.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateI.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateI.getForces()[1], 1e-5);

    // test region II
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(1.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 0.125;
    expectedForce = Vec3(-0.5, 0.0, 0.0);

    State stateII = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateII.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateII.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateII.getForces()[1], 1e-5);

    // test region III
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(2.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 0.0;
    expectedForce = Vec3(0.0, 0.0, 0.0);

    State stateIII = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateIII.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateIII.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateIII.getForces()[1], 1e-5);

    // test region IV
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 0.125;
    expectedForce = Vec3(0.5, 0.0, 0.0);

    State stateIV = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateIV.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateIV.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateIV.getForces()[1], 1e-5);

    // test region V
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(4.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 1.0;
    expectedForce = Vec3(1.0, 0.0, 0.0);

    State stateV = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateV.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateV.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateV.getForces()[1], 1e-5);
}
/* // I can't figure out how to make this work right now
void CudaCalcMeldForceKernel::testEverythingEco() {
  
  
  int counter, counter2, contact_ptr, order;
  float err_tol = 0.0001;
  float dist_sq;
  float x, y, z;
  int src = 0;
  int num_explored = 1;
  int edge_index, head;
  
  // setup system
  const int numParticles = 4;
  System system;
  vector<Vec3> positions(numParticles);
  system.addParticle(1.0);
  system.addParticle(1.0); // add four particles
  system.addParticle(1.0);
  system.addParticle(1.0);

  // setup meld force
  MeldForce* force = new MeldForce();
  int k = 1.0;
  bool doing_eco = true;
  float eco_factor = 0.0;
  float eco_constant = 0.0;
  float eco_linear = 1.0;
  int restIdx0 = force->addDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k, doing_eco, eco_factor, eco_constant, eco_linear, 0, 0);
  int restIdx1 = force->addDistanceRestraint(1, 2, 1.0, 2.0, 3.0, 4.0, k, doing_eco, eco_factor, eco_constant, eco_linear, 0, 0);
  int restIdx2 = force->addDistanceRestraint(2, 3, 1.0, 2.0, 3.0, 4.0, k, doing_eco, eco_factor, eco_constant, eco_linear, 0, 0);
  
  std::vector<int> restIndices(3);
  restIndices[0] = restIdx0;
  restIndices[1] = restIdx1;
  restIndices[2] = restIdx2;
  int groupIdx = force->addGroup(restIndices, 1);
  std::vector<int> groupIndices(1);
  groupIndices[0] = groupIdx;
  force->addCollection(groupIndices, 1);
  system.addForce(force);

  // setup the context
  VerletIntegrator integ(1.0);
  Platform& platform = Platform::getPlatformByName("CUDA");
  Context context(system, integ, platform);

  // test region I
  // set the postitions, compute the forces and energy
  // test to make sure they have the expected values
  positions[0] = Vec3(0.0, 0.0, 0.0);
  positions[1] = Vec3(0.5, 0.0, 0.0);
  positions[2] = Vec3(1.0, 0.0, 0.0);
  positions[3] = Vec3(1.5, 0.0, 0.0);
  
  context.setPositions(positions);

  //float expectedEnergy = 1.0;
  //Vec3 expectedForce = Vec3(-1.0, 0.0, 0.0);

  /*
  State stateI = context.getState(State::Energy | State::Forces);
  ASSERT_EQUAL_TOL(expectedEnergy, stateI.getPotentialEnergy(), 1e-5);
  ASSERT_EQUAL_VEC(expectedForce, stateI.getForces()[0], 1e-5);
  ASSERT_EQUAL_VEC(-expectedForce, stateI.getForces()[1], 1e-5);
  */
  
  // We need to run a series of tests to make sure that everything is behaving like we expect
  /*
  void* test_get_alpha_carbon_posqArgs[] = {
    &cu.getPosq().getDevicePointer(),
    &alphaCarbonPosq->getDevicePointer(),
    &alphaCarbons->getDevicePointer(),
    &numResidues
  };
  cu.executeKernel(test_get_alpha_carbon_posqKernel, test_get_alpha_carbon_posqArgs, numResidues);
  
  //cout << "mark0\n";
  alphaCarbonPosq->download(h_alphaCarbonPosq);
  /*cout << "Alpha Carbon x,y,z:\n";
  for (counter = 0; counter < numResidues; counter++) {
    cout << h_alphaCarbonPosq[counter*3] << "," << h_alphaCarbonPosq[counter*3 + 1] << "," << h_alphaCarbonPosq[counter*3 + 2] << " ";
  }
  cout << "\n";  
  distanceRestContacts->download(h_distanceRestContacts);
  
  /*
  for (counter = 0; counter < numResidues; counter++) {
    contact_ptr = 0;
    for (counter2 = 0; counter2 < numResidues; counter2++) {
      x = h_alphaCarbonPosq[counter*3] - h_alphaCarbonPosq[counter2*3];
      y = h_alphaCarbonPosq[counter*3 + 1] - h_alphaCarbonPosq[counter2*3 + 1];
      z = h_alphaCarbonPosq[counter*3 + 2] - h_alphaCarbonPosq[counter2*3 + 2];
      dist_sq = x*x + y*y + z*z;
      if ( h_distanceRestContacts[numResidues * counter + contact_ptr] == counter2 ) { // then we've hit an edge
        //cout << "Edge between node: " << counter << " and node: " << counter2 << "\n";
        if (dist_sq > ecoCutoff*ecoCutoff && (counter != counter2 - 1 || counter != counter2 + 1)) { // so if the actual distance is greater than expected, then something's wrong
          cout << "ERROR: contact map problem: counter: " << counter << " counter2: " << counter2 << " contact predicted to exist yet distance squared is: " << dist_sq << "\n";
        }
        contact_ptr++; // increment this pointer
      } else { // no contact predicted
        if (dist_sq < ecoCutoff*ecoCutoff && counter != counter2) { // so if the actual distance is less than cutoff, then something's wrong
          cout << "ERROR: contact map problem: counter: " << counter << " counter2: " << counter2 << " contact predicted not to exist yet distance squared is: " << dist_sq << "\n";
        }
      }
    }
  }*
  
  
  // Now test Dijkstra's algorithm results
  for (counter = 0; counter < numResidues; counter++) { // first, initialize the arrays
    h_dijkstra_unexplored[counter] = true;
    h_dijkstra_frontier[counter] = false;
    h_dijkstra_distance[counter] = abs(counter - src);
    h_dijkstra_n_explored[counter] = 0;
    if (counter == src) {
      h_dijkstra_unexplored[counter] = false;
      h_dijkstra_frontier[counter] = true;
      h_dijkstra_distance[counter] = 0;
    }
  }
    
  distanceRestEdgeCounts->download(h_distanceRestEdgeCounts);
  
  counter2 = 0;
  num_explored = 1;
  while ((counter2 <= MAX_ECO_DEPTH) && (num_explored < numResidues)) {
    for (counter = 0; counter < numResidues; counter++) { // save the old arrays
      h_dijkstra_unexplored_old[counter] = h_dijkstra_unexplored[counter];
      h_dijkstra_frontier_old[counter] = h_dijkstra_frontier[counter];
    }
    
    for (counter = 0; counter < numResidues; counter++) { // settle and update
      h_dijkstra_n_explored[counter] = 0;
      if (h_dijkstra_unexplored_old[counter] == true) {
        for (contact_ptr = 0; contact_ptr < h_distanceRestEdgeCounts[counter]; contact_ptr++) {
          edge_index = (numResidues * counter) + contact_ptr;
          head = h_distanceRestContacts[edge_index];  // the index of the node that is leading to this one
          //cout << "edge from node " << counter << " to " << head << "\n";
          if (h_dijkstra_frontier_old[head] == true) { // if the head node is on the frontier, the we need to change our explored status
            h_dijkstra_frontier[counter] = true; // then add myself to the frontier
            h_dijkstra_unexplored[counter] = false; // remove myself from the unexplored
            h_dijkstra_distance[counter] = counter2 + 1; // the number of iterations we've needed to find me is the distance
            num_explored++;
            break;
          }
        }
      }
    }
    counter2++;
  }
  
  // Now rerun the GPU pathfinding algorithm
   void* dijkstra_initializeArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &dijkstra_n_explored->getDevicePointer(),
  &src,
  &INF,
  &numResidues};
  
    /*void* dijkstra_save_oldArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_unexplored_old->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_frontier_old->getDevicePointer(),
  &numResidues};
  *
    void* dijkstra_settle_and_updateArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_unexplored_old->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_frontier_old->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &distanceRestEdgeCounts->getDevicePointer(),
  &distanceRestContacts->getDevicePointer(),
  &dijkstra_n_explored->getDevicePointer(),
  &counter2,
  &numResidues};
  
    /*void* dijkstra_log_reduceArgs[] = {
  &numResidues,
  &dijkstra_n_explored->getDevicePointer(),
  &dijkstra_total->getDevicePointer()};
  *
  cu.executeKernel(dijkstra_initializeKernel, dijkstra_initializeArgs, numResidues);
  counter2 = 0;
  num_explored = 1;
  
  while ((counter2 <= MAX_ECO_DEPTH) && (num_explored < numResidues)) {
    cu.executeKernel(dijkstra_settle_and_updateKernel, dijkstra_settle_and_updateArgs, numResidues);
    //num_explored += h_dijkstra_total[0];
    counter2++;
  }
  
  // by this point, the graphs should be explored
  dijkstra_distance->download(h_dijkstra_distance2);
  for (counter = 0; counter < numResidues; counter++) {
    if (h_dijkstra_distance[counter] != h_dijkstra_distance2[counter] ) { // if these two distances are not equal, then one of the pathfinding algorithms is broken
      cout << "ERROR: pathfinding algorithm discrepancy. Src: 0. Dest: " << counter << " (CPU): " << h_dijkstra_distance[counter] << " (GPU): " << h_dijkstra_distance2[counter] << "\n";
    }
  }
  
} */

void testHyperbolicDistRest() {
    // setup system
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    system.addParticle(1.0);

    // setup meld force
    MeldForce* force = new MeldForce();
    float k = 1.0;
    float asymptote = 3.0;

    int restIdx = force->addHyperbolicDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k, asymptote);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // setup the context
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // There are five regions:
    // I:       r < 1
    // II:  1 < r < 2
    // III: 2 < r < 3
    // IV:  3 < r < 4
    // V:   4 < r

    // test region I
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.5, 0.0, 0.0);
    context.setPositions(positions);

    float expectedEnergy = 1.0;
    Vec3 expectedForce = Vec3(-1.0, 0.0, 0.0);

    State stateI = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateI.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateI.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateI.getForces()[1], 1e-5);

    // test region II
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(1.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 0.125;
    expectedForce = Vec3(-0.5, 0.0, 0.0);

    State stateII = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateII.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateII.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateII.getForces()[1], 1e-5);

    // test region III
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(2.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 0.0;
    expectedForce = Vec3(0.0, 0.0, 0.0);

    State stateIII = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateIII.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateIII.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateIII.getForces()[1], 1e-5);

    // test region IV
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 0.250;
    expectedForce = Vec3(1.0, 0.0, 0.0);

    State stateIV = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateIV.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateIV.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateIV.getForces()[1], 1e-5);

    // test region V
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(4.5, 0.0, 0.0);
    context.setPositions(positions);

    expectedEnergy = 1.666666666;
    expectedForce = Vec3(0.888888888888, 0.0, 0.0);

    State stateV = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, stateV.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce, stateV.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(-expectedForce, stateV.getForces()[1], 1e-5);
}

void testDistRestChangingParameters() {
    // Create particles
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);

    // Define distance restraint
    MeldForce* force = new MeldForce();
    float k = 1.0;
    bool doing_eco = false;
    float eco_factor = 1.0;
    float eco_constant = 0.0;
    float eco_linear = 0.0;
    int res_index1 = 0;
    int res_index2 = 0;
    int restIdx = force->addDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k, doing_eco, eco_factor, eco_constant, eco_linear, res_index1, res_index2);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    float expectedEnergy = 0.125;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

    // Modify the parameters.
    float k2 = 2.0;
    bool doing_eco2 = false;
    float eco_factor2 = 1.0;
    float eco_constant2 = 0.0;
    float eco_linear2 = 0.0;
    int res_index1_2 = 0;
    int res_index2_2 = 0;
    force->modifyDistanceRestraint(0, 0, 1, 1.0, 2.0, 3.0, 4.0, k2, doing_eco2, eco_factor2, eco_constant2, eco_linear2, res_index1_2, res_index2_2);
    force->updateParametersInContext(context);
    state = context.getState(State::Energy);

    // See if the energy is correct after modifying force const.
    expectedEnergy = 0.25;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testTorsRest() {
    // Create particles
    const int numParticles = 4;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(-3.0, -3.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(-3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[2] = Vec3(3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[3] = Vec3(3.0, 3.0, 0.0);

    // Define torsion restraint
    MeldForce* force = new MeldForce();
    float k = 1.0;
    int restIdx = force->addTorsionRestraint(0, 1, 2, 3, 0.0, 0.0, k);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    float expectedEnergy = 16200;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testDistProfileRest() {
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(2.5, 0.0, 0.0);

    MeldForce* force = new MeldForce();
    int nBins = 5;
    int restIdx = 0;
    try {
        std::vector<double> a(nBins);
        for(int i=0; i<a.size(); i++) {
            a[i] = 1.0;
        }
        restIdx = force->addDistProfileRestraint(0, 1, 1.0, 4.0, nBins, a, a, a, a, 1.0);
    }
    catch (std::bad_alloc& ba)
    {
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    
    // See if the energy is correct.
    float expectedEnergy = 75.8565;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testTorsProfileRest() {
    // Create particles
    const int numParticles = 4;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(-3.0, -3.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(-3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[2] = Vec3(3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[3] = Vec3(3.0, 3.0, 0.0);

    // Define torsion restraint
    MeldForce* force = new MeldForce();
    int nBins = 5;
    int restIdx = 0;
    try {
        std::vector<double> a(nBins);
        for(int i=0; i<a.size(); i++) {
            a[i] = 1.0;
        }
        restIdx = force->addTorsProfileRestraint(0, 1, 2, 3, 0, 1, 2, 3, nBins, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, 1.0);
    }
    catch (std::bad_alloc& ba)
    {
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    float expectedEnergy = 1.0;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testGroupSelectsCorrectly() {
    // setup system
    const int numParticles = 3;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);

    // setup meld force
    MeldForce* force = new MeldForce();
    int restIdx1 = force->addDistanceRestraint(0, 1, 0.0, 0.0, 3.0, 999.0, 1.0, false, 1.0, 0.0, 0.0, 0, 0);
    int restIdx2 = force->addDistanceRestraint(1, 2, 0.0, 0.0, 3.0, 999.0, 1.0, false, 1.0, 0.0, 0.0, 0, 0);

    // setup group
    std::vector<int> group(2);
    group[0] = restIdx1;
    group[1] = restIdx2;
    int groupIdx = force->addGroup(group, 1);

    // setup collection
    std::vector<int> collection(1);
    collection[0] = groupIdx;
    force->addCollection(collection, 1);
    system.addForce(force);

    // setup the context
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // set the positions
    // the first has length 4.0
    // the second has length 5.0
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(4.0, 0.0, 0.0);
    positions[2] = Vec3(9.0, 0.0, 0.0);
    context.setPositions(positions);

    // the expected energy is 0.5 * (4 - 3)**2 = 0.5
    float expectedEnergy = 0.5;

    // the force on atom 1 should be
    // f = - k * (4 - 3) = 1.0
    Vec3 expectedForce1 = Vec3(1.0, 0.0, 0.0);
    Vec3 expectedForce2 = -expectedForce1;
    // should be no force on atom 3
    Vec3 expectedForce3 = Vec3(0.0, 0.0, 0.0);

    State state = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce1, state.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(expectedForce2, state.getForces()[1], 1e-5);
    ASSERT_EQUAL_VEC(expectedForce3, state.getForces()[2], 1e-5);
}

void testCollectionSelectsCorrectly() {
    // setup system
    const int numParticles = 3;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);

    // setup meld force
    MeldForce* force = new MeldForce();
    int restIdx1 = force->addDistanceRestraint(0, 1, 0.0, 0.0, 3.0, 999.0, 1.0, false, 1.0, 0.0, 0.0, 0, 0);
    int restIdx2 = force->addDistanceRestraint(1, 2, 0.0, 0.0, 3.0, 999.0, 1.0, false, 1.0, 0.0, 0.0, 0, 0);

    // setup group1
    std::vector<int> group1(1);
    group1[0] = restIdx1;
    int groupIdx1 = force->addGroup(group1, 1);

    // setup group2
    std::vector<int> group2(1);
    group2[0] = restIdx2;
    int groupIdx2 = force->addGroup(group2, 1);

    // setup collection
    std::vector<int> collection(2);
    collection[0] = groupIdx1;
    collection[1] = groupIdx2;
    force->addCollection(collection, 1);
    system.addForce(force);

    // setup the context
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // set the positions
    // the first has length 4.0
    // the second has length 5.0
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(4.0, 0.0, 0.0);
    positions[2] = Vec3(9.0, 0.0, 0.0);
    context.setPositions(positions);

    // the expected energy is 0.5 * (4 - 3)**2 = 0.5
    float expectedEnergy = 0.5;

    // the force on atom 1 should be
    // f = - k * (4 - 3) = 1.0
    Vec3 expectedForce1 = Vec3(1.0, 0.0, 0.0);
    Vec3 expectedForce2 = -expectedForce1;
    // should be no force on atom 3
    Vec3 expectedForce3 = Vec3(0.0, 0.0, 0.0);

    State state = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(expectedForce1, state.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(expectedForce2, state.getForces()[1], 1e-5);
    ASSERT_EQUAL_VEC(expectedForce3, state.getForces()[2], 1e-5);
}


int main(int argc, char* argv[]) {
    try {
        registerMeldCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testDistRest();
        //testEverythingEco();
        testDistRestChangingParameters();
        testHyperbolicDistRest();
        testTorsRest();
        testDistProfileRest();
        testTorsProfileRest();
        testGroupSelectsCorrectly();
        testCollectionSelectsCorrectly();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}

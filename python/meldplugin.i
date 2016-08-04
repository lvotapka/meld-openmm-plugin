/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

%module meldplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"


%include "std_vector.i"

namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
  %template(vectorf) vector<float>;
};


%{
#include "MeldForce.h"
#include "RdcForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <vector>
%}


%pythoncode %{
import simtk.unit as unit
import simtk.openmm as mm
%}

/* include version information */
%pythoncode %{
__version__ = '0.1.3'
%}

/*
    Add units to outputs
*/
%pythonappend MeldPlugin::MeldForce::getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
                                                                float& r4, float& forceConstant, bool& doing_eco, float& eco_factor, float& eco_constant, float& eco_linear, int& res_index1, int& res_index2, int& globalIndex) const %{
   val[2]=unit.Quantity(val[2], unit.nanometer)
   val[3]=unit.Quantity(val[3], unit.nanometer)
   val[4]=unit.Quantity(val[4], unit.nanometer)
   val[5]=unit.Quantity(val[5], unit.nanometer)
   val[6]=unit.Quantity(val[6], unit.kilojoule_per_mole/(unit.nanometer*unit.nanometer))
   val[7]=unit.Quantity(val[7], unit.nanometer)
   val[8]=unit.Quantity(val[8], unit.nanometer)
   val[9]=unit.Quantity(val[9], unit.nanometer)
   val[10]=unit.Quantity(val[10], unit.nanometer)
%}

%pythonappend MeldPlugin::MeldForce::getHyperbolicDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
                                                                          float& r4, float& forceConstant, float& asymptote, int& globalIndex) const %{
   val[2]=unit.Quantity(val[2], unit.nanometer)
   val[3]=unit.Quantity(val[3], unit.nanometer)
   val[4]=unit.Quantity(val[4], unit.nanometer)
   val[5]=unit.Quantity(val[5], unit.nanometer)
   val[6]=unit.Quantity(val[6], unit.kilojoule_per_mole/(unit.nanometer*unit.nanometer))
   val[7]=unit.Quantity(val[7], unit.kilojoule_per_mole)
%}

%pythonappend MeldPlugin::MeldForce::getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int&atom4,
                                                               float& phi, float& deltaPhi, float& forceConstant, int& globalIndex) const %{
    val[4] = unit.Quantity(val[4], unit.degree)
    val[5] = unit.Quantity(val[5], unit.degree)
    val[6] = unit.Quantity(val[6], unit.kilojoule_per_mole / (unit.degree * unit.degree))
%}

%pythonappend MeldPlugin::MeldForce::getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
                                                                   int& nBins, std::vector<double>& a0, std::vector<double>& a1,
                                                                   std::vector<double>& a2, std::vector<double>& a3,
                                                                   float& scaleFactor, int& globalIndex) const %{
    val[2] = unit.Quantity(val[2], unit.nanometer)
    val[3] = unit.Quantity(val[3], unit.nanometer)
%}

/*
  The actual routines to wrap are below
*/
namespace MeldPlugin {

    class MeldForce : public OpenMM::Force {
    public:
        MeldForce();

        void updateParametersInContext(OpenMM::Context& context);

        int getNumDistRestraints() const;
        
        int setEcoCutoff(float eco_cut_value);
        
        int setEcoOutputFreq(int eco_output_freq_value);
        
        int setPrintAvgEco(bool print_avg_eco_value);
        
        int setPrintEcoValueArray(bool print_eco_value_array_value);
        
        int setCurrentReplicaIndex(int current_replica_index_value);
        
        int setStartingReplicaIndex(int starting_replica_index_value);
        
        int setAlphaCarbonVector(std::vector< int > alpha_carbon_vector);
        
        int uploadCartProfileRestCoeffs(std::vector< float > globalCartProfileRestCoeffsArg);
        
        int setDistRestSortedVector(std::vector< int > dist_rest_sorted_vector);

        int getNumTorsionRestraints() const;

        int getNumDistProfileRestraints() const;

        int getNumDistProfileRestParams() const;

        int getNumTorsProfileRestraints() const;

        int getNumTorsProfileRestParams() const;
        
        int getNumCartProfileRestraints() const;
        
        int getNumCartProfileRestParams() const;

        int getNumTotalRestraints() const;

        int getNumGroups() const;

        int getNumCollections() const;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply float& OUTPUT {float& r1};
        %apply float& OUTPUT {float& r2};
        %apply float& OUTPUT {float& r3};
        %apply float& OUTPUT {float& r4};
        %apply float& OUTPUT {float& forceConstant};
        %apply bool& OUTPUT {bool& doing_eco};
        %apply float& OUTPUT {float& eco_factor};
        %apply float& OUTPUT {float& eco_constant};
        %apply float& OUTPUT {float& eco_linear};
        %apply int& OUTPUT {int& res_index1};
        %apply int& OUTPUT {int& res_index2};
        %apply int& OUTPUT {int& globalIndex};
        void getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
                float& r4, float& forceConstant, bool& doing_eco, float& eco_factor, float& eco_constant, float& eco_linear, int& res_index1, int& res_index2, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear float& r1;
        %clear float& r2;
        %clear float& r3;
        %clear float& r4;
        %clear float& forceConstant;
        %clear bool& doing_eco;
        %clear float& eco_factor;
        %clear float& eco_constant;
        %clear float& eco_linear;
        %clear int& res_index1;
        %clear int& res_index2;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply int& OUTPUT {int& atom3};
        %apply int& OUTPUT {int& atom4};
        %apply float& OUTPUT {float& phi};
        %apply float& OUTPUT {float& deltaPhi};
        %apply float& OUTPUT {float& forceConstant};
        %apply int& OUTPUT {int& globalIndex};
        void getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int&atom4,
                float& phi, float& deltaPhi, float& forceConstant, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear int& atom3;
        %clear int& atom4;
        %clear float& phi;
        %clear float& deltaPhi;
        %clear float& forceConstant;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply float& OUTPUT {float& rMin};
        %apply float& OUTPUT {float& rMax};
        %apply int& OUTPUT {int& nBins};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a0};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a1};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a2};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a3};
        %apply float& OUTPUT {float& scaleFactor};
        %apply int& OUTPUT {int& globalIndex};
        void getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
                int& nBins, std::vector<double>& a0, std::vector<double>& a1, std::vector<double>& a2,
                std::vector<double>& a3, float& scaleFactor, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear float& rMin;
        %clear float& rMax;
        %clear int& nBins;
        %clear vector<double>& a0;
        %clear std::vector<double>& a1;
        %clear std::vector<double>& a2;
        %clear std::vector<double>& a3;
        %clear float& scaleFactor;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply int& OUTPUT {int& atom3};
        %apply int& OUTPUT {int& atom4};
        %apply int& OUTPUT {int& atom5};
        %apply int& OUTPUT {int& atom6};
        %apply int& OUTPUT {int& atom7};
        %apply int& OUTPUT {int& atom8};
        %apply int& OUTPUT {int& nBins};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a0};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a1};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a2};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a3};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a4};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a5};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a6};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a7};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a8};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a9};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a10};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a11};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a12};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a13};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a14};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a15};
        %apply float& OUTPUT {float& scaleFactor};
        %apply int& OUTPUT {int& globalIndex};
        void getTorsProfileRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4,
                int& atom5, int& atom6, int& atom7, int& atom8, int& nBins,
                std::vector<double>&  a0, std::vector<double>&  a1, std::vector<double>&  a2,
                std::vector<double>&  a3, std::vector<double>&  a4, std::vector<double>&  a5,
                std::vector<double>&  a6, std::vector<double>&  a7, std::vector<double>&  a8,
                std::vector<double>&  a9, std::vector<double>& a10, std::vector<double>& a11,
                std::vector<double>& a12, std::vector<double>& a13, std::vector<double>& a14,
                std::vector<double>& a15, float& scaleFactor, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear int& atom3;
        %clear int& atom4;
        %clear int& atom5;
        %clear int& atom6;
        %clear int& atom7;
        %clear int& atom8;
        %clear int& nBins;
        %clear std::vector<double>& a0;
        %clear std::vector<double>& a1;
        %clear std::vector<double>& a2;
        %clear std::vector<double>& a3;
        %clear std::vector<double>& a4;
        %clear std::vector<double>& a5;
        %clear std::vector<double>& a6;
        %clear std::vector<double>& a7;
        %clear std::vector<double>& a8;
        %clear std::vector<double>& a9;
        %clear std::vector<double>& a10;
        %clear std::vector<double>& a11;
        %clear std::vector<double>& a12;
        %clear std::vector<double>& a13;
        %clear std::vector<double>& a14;
        %clear std::vector<double>& a15;
        %clear float& scaleFactor;
        %clear int& globalIndex;
        
        %apply int& OUTPUT {int& atom};
        %apply int& OUTPUT {int& startingCoeff};
        %apply int& OUTPUT {int& dimx};
        %apply int& OUTPUT {int& dimy};
        %apply int& OUTPUT {int& dimz};
        %apply float& OUTPUT {float& resx};
        %apply float& OUTPUT {float& resy};
        %apply float& OUTPUT {float& resz};
        %apply float& OUTPUT {float& origx};
        %apply float& OUTPUT {float& origy};
        %apply float& OUTPUT {float& origz};
        %apply float& OUTPUT {float& scaleFactor};
        %apply int& OUTPUT {int& globalIndex};
        void getCartProfileRestraintParams(int index, int& atom, int& startingCoeff,
            int& dimx, int& dimy, int& dimz, float& resx, float& resy, float& resz, 
            float& origx, float& origy, float& origz, float& scaleFactor, int& globalIndex) const;
        %clear int& atom;
        %clear int& startingCoeff;
        %clear int& dimx;
        %clear int& dimy;
        %clear int& dimz;
        %clear float& resx;
        %clear float& resy;
        %clear float& resz;
        %clear float& origx;
        %clear float& origy;
        %clear float& origz;
        %clear float& scaleFactor;
        %clear int& globalIndex;
        
        %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
        %apply int& OUTPUT {int& numActive};
        void getGroupParams(int index, std::vector<int>& indices, int& numActive) const;
        %clear std::vector<int>& indices;
        %clear int& numActive;

        %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
        %apply int& OUTPUT {int& numActive};
        void getCollectionParams(int index, std::vector<int>& indices, int& numActive) const;
        %clear std::vector<int>& indices;
        %clear int& numActive;

        int addDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant, bool doing_eco, float eco_factor, float eco_constant, float eco_linear, int res_index1, int res_index2);

        void modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
                float r4, float force_constant, bool doing_eco, float eco_factor, float eco_constant, float eco_linear, int res_index1, int res_index2);

        int addHyperbolicDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant, float asymptote);

        void modifyHyperbolicDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
                float r4, float force_constant, float asymptote);

        int addTorsionRestraint(int atom1, int atom2, int atom3, int atom4, float phi, float deltaPhi, float forceConstant);

        void modifyTorsionRestraint(int index, int atom1, int atom2, int atom3, int atom4, float phi,
                float deltaPhi, float forceConstant);

        int addDistProfileRestraint(int atom1, int atom2, float rMin, float rMax, int nBins, std::vector<double> a0,
                std::vector<double> a1, std::vector<double> a2, std::vector<double> a3, float scaleFactor);

        void modifyDistProfileRestraint(int index, int atom1, int atom2, float rMin, float rMax, int nBins,
                std::vector<double> a0, std::vector<double> a1, std::vector<double> a2, std::vector<double> a3,
                float scaleFactor);

        int addTorsProfileRestraint(int atom1, int atom2, int atom3, int atom4,
                int atom5, int atom6, int atom7, int atom8, int nBins,
                std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
                std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
                std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
                std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
                std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                std::vector<double> a15, float scaleFactor);

        void modifyTorsProfileRestraint(int index, int atom1, int atom2, int atom3, int atom4,
                int atom5, int atom6, int atom7, int atom8, int nBins,
                std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
                std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
                std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
                std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
                std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                std::vector<double> a15, float scaleFactor);
                
        int addCartProfileRestraint(int atom, int startingCoeff,
                int dimx, int dimy, int dimz, float resx, float resy, float resz, 
                float origx, float origy, float origz, float scaleFactor);
                
        void modifyCartProfileRestraint(int index, int atom, int startingCoeff,
            int dimx, int dimy, int dimz, float resx, float resy, float resz, 
            float origx, float origy, float origz, float scaleFactor);

        int addGroup(std::vector<int> restraint_indices, int n_active);

        int addCollection(std::vector<int> group_indices, int n_active);
    };

    class RdcForce : public OpenMM::Force {
    public:
        RdcForce();

        void updateParametersInContext(OpenMM::Context& context);

        int getNumExperiments() const;

        int getNumRestraints(int experiment) const;

        int getNumTotalRestraints() const;

        int addExperiment(std::vector<int> rdcIndices);

        int addRdcRestraint(int particle1, int particle2, float kappa, float dObs, float tolerance,
                float force_const, float weight);

        void updateRdcRestraint(int index, int particle1, int particle2, float kappa, float dObs,
                float tolerance, float force_const, float weight);

        %apply std::vector<int>& OUTPUT {std::vector<int>& restraints};
        void getExperimentInfo(int index, std::vector<int>& restraints) const;
        %clear std::vector<int> & restraints;

        %apply int& OUTPUT {int & particle1};
        %apply int& OUTPUT {int & particle2};
        %apply float& OUTPUT {float & kappa};
        %apply float& OUTPUT {float & dObs};
        %apply float& OUTPUT {float & tolerance};
        %apply float& OUTPUT {float & force_const};
        %apply float& OUTPUT {float & weight};
        %apply int& OUTPUT {int & globalIndex};
        void getRdcRestraintInfo(int index, int& particle1, int& partcile2, float& kappa,
                float& dObs, float& tolerance, float& force_const, float& weight,
                int& globalIndex) const;
        %clear int & particle1;
        %clear int & particle2;
        %clear float & kappa;
        %clear float & dObs;
        %clear float & tolerance;
        %clear float & force_const;
        %clear float & weight;
        %clear int & globalIndex;
    };
}

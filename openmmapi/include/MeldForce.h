#ifndef OPENMM_MELD_FORCE_H_
#define OPENMM_MELD_FORCE_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportMeld.h"
#include <map>
#include <vector>

namespace MeldPlugin {

/**
 * This is the MELD Force.
 */

class OPENMM_EXPORT_MELD MeldForce : public OpenMM::Force {

public:
    /**
     * Default constructor
     */
     MeldForce();

    /**
     * Update the per-restraint parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call modifyDistanceRestaint(), modifyTorsionRestraint(), modifyDistProfileRestraint(),
     * or modifyTorsProfileRestraint() to modify the parameters of a restraint, then call updateParametersInContext()
     * to copy them over to the Context.
     * 
     * This method has several limitations.  The only information it updates is the values of per-restraint parameters.
     * All other aspects of the Force (such as the energy function) are unaffected and can only be changed by reinitializing
     * the Context.  The set of particles involved in a restraint cannot be changed, nor can new restraints be added.
     */
    void updateParametersInContext(OpenMM::Context& context);

    /**
     * @return The number of distance restraints.
     */
    int getNumDistRestraints() const;
   
    /**
     * @return The number of torsion restraints.
     */
    int getNumTorsionRestraints() const;
   
    /**
     * @return The number of distance profile restraints.
     */
    int getNumDistProfileRestraints() const;
   
    /**
     * @return The number of distance profile restraint parameters.
     */
    int getNumDistProfileRestParams() const;

    /**
     * @return The number of torsion profile restraints.
     */
    int getNumTorsProfileRestraints() const;

    /**
     * @return The number of torsion profile restraint parameters.
     */
    int getNumTorsProfileRestParams() const;

    /**
     * @return The total number of distance and torsion restraints.
     */
    int getNumTotalRestraints() const;

    /**
     * @return The number of restraint groups.
     */
    int getNumGroups() const;

    /**
     * @return The number of collections of restraint groups.
     */
    int getNumCollections() const;

    /**
     * Get the parameters for a distance restraint. See addDistanceRestraint()
     * for more details about the parameters.
     * 
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     * @param globalIndex  the global index of the restraint
     */
    void getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
            float& r4, float& forceConstant, int& globalIndex) const;

    /**
     * Get the parameters for a torsion restraint. See addTorsionRestraint() for
     * more details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param phi  the equilibrium torsion (degrees)
     * @param deltaPhi  the deltaPhi parameter (degrees)
     * @param forceConstant  the force constant
     * @param globalIndex  the global index of the restraint
     */
    void getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int&atom4,
            float& phi, float& deltaPhi, float& forceConstant, int& globalIndex) const;

    /**
     * Get the parameters for a distance profile restraint. See addDistProfileRestraint()
     * for more details about the parameters.
     * 
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param rMin  the lower bound of the restraint
     * @param rMax  the upper bound of the restraint
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,2,3)
     * @param scaleFactor  the scale factor
     * @param globalIndex  the global index of the restraint
     */
    void getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
            int& nBins, std::vector<double>& a0, std::vector<double>& a1, std::vector<double>& a2,
            std::vector<double>& a3, float& scaleFactor, int& globalIndex) const;

    /**
     * Get the parameters for a torsion profile restraint.
     * 
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param atom5  the fifth atom
     * @param atom6  the sixth atom
     * @param atom7  the seventh atom
     * @param atom8  the eighth atom
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,...,14,15)
     * @param scaleFactor  the scale factor
     * @param globalIndex  the global index of the restraint
     */
    void getTorsProfileRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4,
            int& atom5, int& atom6, int& atom7, int& atom8, int& nBins,
            std::vector<double>&  a0, std::vector<double>&  a1, std::vector<double>&  a2,
            std::vector<double>&  a3, std::vector<double>&  a4, std::vector<double>&  a5,
            std::vector<double>&  a6, std::vector<double>&  a7, std::vector<double>&  a8,
            std::vector<double>&  a9, std::vector<double>& a10, std::vector<double>& a11,
            std::vector<double>& a12, std::vector<double>& a13, std::vector<double>& a14,
            std::vector<double>& a15, float& scaleFactor, int& globalIndex) const;

    /**
     * Get the parameters for a group of restraints.
     * 
     * @param index  the index of the group
     * @param indices  the indices of the restraints in the group
     * @param numActive  the number of active restraints in the group
     */
    void getGroupParams(int index, std::vector<int>& indices, int& numActive) const;

    /**
     * Get the parameters for a collection of restraint groups.
     *
     * @param index  the index of the collection
     * @param indices  the indices of the groups in the collection
     * @param numActive  the number of active groups in the collection
     */
    void getCollectionParams(int index, std::vector<int>& indices, int& numActive) const;

    /**
     * Create a new distance restraint.
     * There are five regions:
     *
     * I:    r < r1
     *
     * II:  r1 < r < r2
     *
     * III: r2 < r < r3
     *
     * IV:  r3 < r < r4
     *
     * V:   r4 < r
     *
     * @param particle1  the first atom
     * @param particle2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     * @return the index of the restraint that was created
     */
    int addDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
            float force_constant);

    /**
     * Modify an existing distance restraint. See addDistanceRestraint() for more
     * details about the parameters.
     * 
     * @param index  the index of the restraint
     * @param particle1  the first atom
     * @param particle2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     */
    void modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
            float r4, float force_constant);

    /**
     * Create a new torsion restraint.
     * 
     * If (x - phi) < -deltaPhi:
     *    E = 1/2 * forceConstant * (x - phi + deltaPhi)^2
     *
     * Else if (x - phi) > deltaPhi:
     *    E = 1/2 * forceConstant * (x - phi - deltaPhi)^2
     *
     * Else:
     *    E = 0
     *
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param phi  the equilibrium torsion (degrees)
     * @param deltaPhi  the deltaPhi parameter (degrees)
     * @param forceConstant  the force constant
     * @return the index of the restraint that was created
     */
    int addTorsionRestraint(int atom1, int atom2, int atom3, int atom4, float phi, float deltaPhi, float forceConstant);

    /**
     * Modify an existing torsion restraint. See addTorsionRestraint() for more
     * details about the parameters.
     * 
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param phi  the equilibrium torsion (degrees)
     * @param deltaPhi  the deltaPhi parameter (degrees)
     * @param forceConstant  the force constant
     */
    void modifyTorsionRestraint(int index, int atom1, int atom2, int atom3, int atom4, float phi,
            float deltaPhi, float forceConstant);

    /**
     * Create a new distance profile restraint.
     *
     * bin = floor( (r - rMin) / (rMax - rMin) * nBins) )
     *
     * binWidth = (rMax - rMin) / nBins
     *
     * t = (r - bin * binWidth + rMin) / binWidth;
     *
     * E = scaleFactor * (a0 + a1 * t + a2 * t^2 + a3 * t^3)
     * 
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param rMin  the lower bound of the restraint
     * @param rMax  the upper bound of the restraint
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,2,3)
     * @param scaleFactor  the scale factor
     * @return the index of the restraint that was created
     */
    int addDistProfileRestraint(int atom1, int atom2, float rMin, float rMax, int nBins, std::vector<double> a0,
            std::vector<double> a1, std::vector<double> a2, std::vector<double> a3, float scaleFactor);

    /**
     * Modify an existing distance profile restraint. See addDistProfileRestraint()
     * for more details about the parameters.
     * 
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param rMin  the lower bound of the restraint
     * @param rMax  the upper bound of the restraint
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,2,3)
     * @param scaleFactor  the scale factor
     */
    void modifyDistProfileRestraint(int index, int atom1, int atom2, float rMin, float rMax, int nBins,
            std::vector<double> a0, std::vector<double> a1, std::vector<double> a2, std::vector<double> a3,
            float scaleFactor);

    /**
     * Create a new torsion profile restraint.
     * 
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param atom5  the fifth atom
     * @param atom6  the sixth atom
     * @param atom7  the seventh atom
     * @param atom8  the eighth atom
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,...,14,15)
     * @param scaleFactor  the scale factor
     * @return the index of the restraint that was created
     */
    int addTorsProfileRestraint(int atom1, int atom2, int atom3, int atom4,
            int atom5, int atom6, int atom7, int atom8, int nBins,
            std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
            std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
            std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
            std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
            std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
            std::vector<double> a15, float scaleFactor);

    /**
     * Modify an existing torsion profile restraint.
     * 
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param atom5  the fifth atom
     * @param atom6  the sixth atom
     * @param atom7  the seventh atom
     * @param atom8  the eighth atom
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,...,14,15)
     * @param scaleFactor  the scale factor
     */
    void modifyTorsProfileRestraint(int index, int atom1, int atom2, int atom3, int atom4,
            int atom5, int atom6, int atom7, int atom8, int nBins,
            std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
            std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
            std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
            std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
            std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
            std::vector<double> a15, float scaleFactor);

    /**
     * Create a new group of restraints.
     * 
     * @param restraint_indices  the indices of the restraints in the group
     * @param n_active  the number of active restraints in the group
     * @return the index of the group that was created
     */
    int addGroup(std::vector<int> restraint_indices, int n_active);

    /**
     * Create a new collection of restraint groups.
     * 
     * @param group_indices  the indices of the groups in the collection
     * @param n_active  the number of active groups in the collection
     * @return the index of the collection that was created
     */
    int addCollection(std::vector<int> group_indices, int n_active);

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class TorsionRestraintInfo;
    class DistanceRestraintInfo;
    class DistProfileRestraintInfo;
    class TorsProfileRestraintInfo;
    class GroupInfo;
    class CollectionInfo;
    int n_restraints;
    std::vector<DistanceRestraintInfo> distanceRestraints;
    std::vector<TorsionRestraintInfo> torsions;
    std::vector<DistProfileRestraintInfo> distProfileRestraints;
    std::vector<TorsProfileRestraintInfo> torsProfileRestraints;
    std::vector<GroupInfo> groups;
    std::vector<CollectionInfo> collections;

    class DistanceRestraintInfo {
    public:
        int particle1, particle2;
        float r1, r2, r3, r4, force_constant;
        int global_index;

        DistanceRestraintInfo() {
            particle1 = particle2    = -1;
            force_constant = 0.0;
            r1 = r2 = r3 = r4 = 0.0;
            global_index = -1;
        }

        DistanceRestraintInfo(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant, int global_index) : particle1(particle1), particle2(particle2), r1(r1),
                                                            r2(r2), r3(r3), r4(r4), force_constant(force_constant),
                                                            global_index(global_index) {
        }
    };

    class TorsionRestraintInfo {
    public:
        int atom1, atom2, atom3, atom4;
        float phi, deltaPhi, forceConstant;
        int globalIndex;

        TorsionRestraintInfo() {
            atom1 = atom2 = atom3 = atom4 = -1;
            phi = 0;
            deltaPhi = 0;
            forceConstant = 0;
            globalIndex =  -1;
        }

        TorsionRestraintInfo(int atom1, int atom2, int atom3, int atom4, float phi, float deltaPhi,
                               float forceConstant, int globalIndex) :
            atom1(atom1), atom2(atom2), atom3(atom3), atom4(atom4), phi(phi), deltaPhi(deltaPhi),
            forceConstant(forceConstant), globalIndex(globalIndex) {
        }
    };

    class DistProfileRestraintInfo {
    public:
        int atom1, atom2, nBins;
        float rMin, rMax, scaleFactor;
        std::vector<double> a0;
        std::vector<double> a1;
        std::vector<double> a2;
        std::vector<double> a3;
        int globalIndex;

        DistProfileRestraintInfo() {
            atom1 = atom2 = -1;
            nBins = 0;
            rMin = rMax = scaleFactor = -1.0;
            globalIndex = -1;
        }

        DistProfileRestraintInfo(int atom1, int atom2, float rMin, float rMax, int nBins,
                std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
                std::vector<double> a3, float scaleFactor, int globalIndex) :
            atom1(atom1), atom2(atom2), nBins(nBins), rMin(rMin), rMax(rMax), scaleFactor(scaleFactor),
            globalIndex(globalIndex), a0(a0), a1(a1), a2(a2), a3(a3) {
            }
    };

    class TorsProfileRestraintInfo {
    public:
        int atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, globalIndex;
        float scaleFactor;
        std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;

        TorsProfileRestraintInfo() {
            atom1 = atom2 = atom3 = atom4 = atom5 = atom6 = atom7 = atom8 = nBins = globalIndex = -1;
            scaleFactor = 0;
        }

        TorsProfileRestraintInfo(int atom1, int atom2, int atom3, int atom4,
                int atom5, int atom6, int atom7, int atom8, int nBins,
                std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
                std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
                std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
                std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
                std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                std::vector<double> a15, float scaleFactor, int globalIndex) :
            atom1(atom1), atom2(atom2), atom3(atom3), atom4(atom4),
            atom5(atom5), atom6(atom6), atom7(atom7), atom8(atom8), nBins(nBins),
            a0(a0), a1(a1), a2(a2), a3(a3), a4(a4), a5(a5), a6(a6), a7(a7),
            a8(a8), a9(a9), a10(a10), a11(a11), a12(a12), a13(a13), a14(a14), a15(a15),
            scaleFactor(scaleFactor), globalIndex(globalIndex) {
            }
    };

    class GroupInfo {
    public:
        std::vector<int> restraint_indices;
        int n_active;

        GroupInfo(): n_active(0) {
        }

        GroupInfo(std::vector<int> restraint_indices, int n_active):
            restraint_indices(restraint_indices), n_active(n_active) {
        }
    };

    class CollectionInfo {
    public:
        std::vector<int> group_indices;
        int n_active;

        CollectionInfo(): n_active(0) {
        }

        CollectionInfo(std::vector<int> group_indices, int n_active) :
            group_indices(group_indices), n_active(n_active) {
        }
    };
};

} // namespace MeldPlugin

#endif /*OPENMM_MELD_FORCE_H_*/

/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(gcmc/vp,FixGCMCVP);
// clang-format on
#else

#ifndef LMP_FIX_GCMC_VP_H
#define LMP_FIX_GCMC_VP_H

#include "fix_gcmc.h"

namespace LAMMPS_NS {

class FixGCMCVP : public FixGCMC, public Fix {
 public:
  FixGCMCVP(class LAMMPS *, int, char **);

  void init() override;            // Finished
  void pre_exchange() override;    // Finished
  void update_gas_atoms_list();    // Finished

  // Atomic
  void attempt_atomic_deletion();     // Finished
  void attempt_atomic_insertion();    // Finished

  // Atomic Full
  void attempt_atomic_deletion_full();    // Finished
  void attempt_atomic_insertion_full();

  // Forbidden Moves
  void attempt_atomic_translation_full() {}      // Finished
  void attempt_molecule_translation_full() {}    // Finished
  void attempt_molecule_rotation_full() {}       // Finished
  void attempt_molecule_deletion_full() {}       // Finished
  void attempt_molecule_insertion_full() {}      // Finished

  double compute_vector(int) override;    // Finished

 private:
  // VP requirements
  int pairflag;          // Pair style flag, 0=lj/cut, 1=SW
  double energyout;      // Total energy change for compute vector
  class Pair *pairsw;    // Pair class for Stw_GCMC

  int molecule_group, molecule_group_bit;
  int molecule_group_inversebit;
  int exclusion_group, exclusion_group_bit;
  int ngcmc_type, nevery, seed;
  int ncycles, nexchanges, nmcmoves;
  double patomtrans, pmoltrans, pmolrotate, pmctot;
  int ngas;                // # of gas atoms on all procs
  int ngas_local;          // # of gas atoms on this proc
  int ngas_before;         // # of gas atoms on procs < this proc
  int exchmode;            // exchange ATOM or MOLECULE
  int movemode;            // move ATOM or MOLECULE
  class Region *region;    // gcmc region
  char *idregion;          // gcmc region id
  bool pressure_flag;      // true if user specified reservoir pressure
  bool charge_flag;        // true if user specified atomic charge
  bool full_flag;          // true if doing full system energy calculations

  int natoms_per_molecule;    // number of atoms in each inserted molecule
  int nmaxmolatoms;           // number of atoms allocated for molecule arrays

  int groupbitall;            // group bitmask for inserted atoms
  int ngroups;                // number of group-ids for inserted atoms
  char **groupstrings;        // list of group-ids for inserted atoms
  int ngrouptypes;            // number of type-based group-ids for inserted atoms
  char **grouptypestrings;    // list of type-based group-ids for inserted atoms
  int *grouptypebits;         // list of type-based group bitmasks
  int *grouptypes;            // list of type-based group types
  double ntranslation_attempts;
  double ntranslation_successes;
  double nrotation_attempts;
  double nrotation_successes;
  double ndeletion_attempts;
  double ndeletion_successes;
  double ninsertion_attempts;
  double ninsertion_successes;

  int gcmc_nmax;
  int max_region_attempts;
  double gas_mass;
  double reservoir_temperature;
  double tfac_insert;
  double chemical_potential;
  double displace;
  double max_rotation_angle;
  double beta, zz, sigma, volume;
  double pressure, fugacity_coeff, charge;
  double xlo, xhi, ylo, yhi, zlo, zhi;
  double region_xlo, region_xhi, region_ylo, region_yhi, region_zlo, region_zhi;
  double region_volume;
  double energy_stored;    // full energy of old/current configuration
  double *sublo, *subhi;
  int *local_gas_list;
  double **cutsq;
  double **molcoords;
  double *molq;
  imageint *molimage;
  imageint imagezero;
  double overlap_cutoffsq;    // square distance cutoff for overlap
  int overlap_flag;
  int max_ngas;
  int min_ngas;

  double energy_intra;

  class Pair *pair;

  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Atom *model_atom;

  class Molecule **onemols;
  int imol, nmol;
  class Fix *fixrigid, *fixshake;
  int rigidflag, shakeflag;
  char *idrigid, *idshake;
  int triclinic;    // 0 = orthog box, 1 = triclinic

  class Compute *c_pe;
  void options(int, char **);
};

}    // namespace LAMMPS_NS

#endif
#endif

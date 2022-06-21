// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier, Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "fix_gcmc_vp.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "region.h"
#include "update.h"

// GCMC_VP specific
#include "pair_hybrid_overlay.h"    // added by Jibao
#include "pair_sw.h"        // added by Jibao

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// large energy value used to signal overlap

#define MAXENERGYSIGNAL 1.0e100

// this must be lower than MAXENERGYSIGNAL
// by a large amount, so that it is still
// less than total energy when negative
// energy contributions are added to MAXENERGYSIGNAL

#define MAXENERGYTEST 1.0e50

enum{EXCHATOM,EXCHMOL}; // exchmode
enum{NONE,MOVEATOM,MOVEMOL}; // movemode

/* ---------------------------------------------------------------------- */

FixGCMCVP::FixGCMCVP(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg),
  region(nullptr), idregion(nullptr), full_flag(false), groupstrings(nullptr),
  grouptypestrings(nullptr), grouptypebits(nullptr), grouptypes(nullptr), local_gas_list(nullptr),
  molcoords(nullptr), molq(nullptr), molimage(nullptr), random_equal(nullptr), random_unequal(nullptr),
  fixrigid(nullptr), fixshake(nullptr), idrigid(nullptr), idshake(nullptr)
{
  if (narg < 11) error->all(FLERR,"Illegal fix gcmc command");

  if (atom->molecular == Atom::TEMPLATE)
    error->all(FLERR,"Fix gcmc does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 9;  // Increased from 8 to 9 to include energyout in compute vector
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  ngroups = 0;
  ngrouptypes = 0;

  // VP Required
  pairflag = 0;

  // required args

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  nexchanges = utils::inumeric(FLERR,arg[4],false,lmp);
  nmcmoves = utils::inumeric(FLERR,arg[5],false,lmp);
  ngcmc_type = utils::inumeric(FLERR,arg[6],false,lmp);
  seed = utils::inumeric(FLERR,arg[7],false,lmp);
  reservoir_temperature = utils::numeric(FLERR,arg[8],false,lmp);
  chemical_potential = utils::numeric(FLERR,arg[9],false,lmp);
  displace = utils::numeric(FLERR,arg[10],false,lmp);

  if (nevery <= 0) error->all(FLERR,"Illegal fix gcmc command");
  if (nexchanges < 0) error->all(FLERR,"Illegal fix gcmc command");
  if (nmcmoves < 0) error->all(FLERR,"Illegal fix gcmc command");
  if (seed <= 0) error->all(FLERR,"Illegal fix gcmc command");
  if (reservoir_temperature < 0.0) 
    error->all(FLERR,"Illegal fix gcmc command");
  if (displace < 0.0) error->all(FLERR,"Illegal fix gcmc command");

  // read options from end of input line

  options(narg-11,&arg[11]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // error checks on region and its extent being inside simulation box

  region_xlo = region_xhi = region_ylo = region_yhi =
    region_zlo = region_zhi = 0.0;
  if (region) {
    if (region->bboxflag == 0)
      error->all(FLERR,"Fix gcmc region does not support a bounding box");
    if (region->dynamic_check())
      error->all(FLERR,"Fix gcmc region cannot be dynamic");

    region_xlo = region->extent_xlo;
    region_xhi = region->extent_xhi;
    region_ylo = region->extent_ylo;
    region_yhi = region->extent_yhi;
    region_zlo = region->extent_zlo;
    region_zhi = region->extent_zhi;

    if (region_xlo < domain->boxlo[0] || region_xhi > domain->boxhi[0] ||
        region_ylo < domain->boxlo[1] || region_yhi > domain->boxhi[1] ||
        region_zlo < domain->boxlo[2] || region_zhi > domain->boxhi[2])
      error->all(FLERR,"Fix gcmc region extends outside simulation box");

    // estimate region volume using MC trials

    double coord[3];
    int inside = 0;
    int attempts = 10000000;
    for (int i = 0; i < attempts; i++) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      if (region->match(coord[0],coord[1],coord[2]) != 0)
        inside++;
    }

    double max_region_volume = (region_xhi - region_xlo) *
      (region_yhi - region_ylo) * (region_zhi - region_zlo);

    region_volume = max_region_volume * static_cast<double>(inside) / static_cast<double>(attempts);
  }

  // error check and further setup for exchmode = EXCHMOL

  if (exchmode == EXCHMOL) {
    if (onemols[imol]->xflag == 0)
      error->all(FLERR,"Fix gcmc molecule must have coordinates");
    if (onemols[imol]->typeflag == 0)
      error->all(FLERR,"Fix gcmc molecule must have atom types");
    if (ngcmc_type != 0)
      error->all(FLERR,"Atom type must be zero in fix gcmc mol command");
    if (onemols[imol]->qflag == 1 && atom->q == nullptr)
      error->all(FLERR,"Fix gcmc molecule has charges, but atom style does not");

    if (atom->molecular == Atom::TEMPLATE && onemols != atom->avec->onemols)
      error->all(FLERR,"Fix gcmc molecule template ID must be same "
                 "as atom_style template ID");
    onemols[imol]->check_attributes(0);
  }

  if (charge_flag && atom->q == nullptr)
    error->all(FLERR,"Fix gcmc atom has charge, but atom style does not");

  if (rigidflag && exchmode == EXCHATOM)
    error->all(FLERR,"Cannot use fix gcmc rigid and not molecule");
  if (shakeflag && exchmode == EXCHATOM)
    error->all(FLERR,"Cannot use fix gcmc shake and not molecule");
  if (rigidflag && shakeflag)
    error->all(FLERR,"Cannot use fix gcmc rigid and shake");
  if (rigidflag && (nmcmoves > 0))
    error->all(FLERR,"Cannot use fix gcmc rigid with MC moves");
  if (shakeflag && (nmcmoves > 0))
    error->all(FLERR,"Cannot use fix gcmc shake with MC moves");

  // setup of array of coordinates for molecule insertion
  // also used by rotation moves for any molecule

  if (exchmode == EXCHATOM) natoms_per_molecule = 1;
  else natoms_per_molecule = onemols[imol]->natoms;
  nmaxmolatoms = natoms_per_molecule;
  grow_molecule_arrays(nmaxmolatoms);

  // compute the number of MC cycles that occur nevery timesteps

  ncycles = nexchanges + nmcmoves;

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  ntranslation_attempts = 0.0;
  ntranslation_successes = 0.0;
  nrotation_attempts = 0.0;
  nrotation_successes = 0.0;
  ndeletion_attempts = 0.0;
  ndeletion_successes = 0.0;
  ninsertion_attempts = 0.0;
  ninsertion_successes = 0.0;

  gcmc_nmax = 0;
  local_gas_list = nullptr;
  
  // VP Specific -- TODO I don't know why these values for the coeffs are chosen
  if(pairflag==1){                 //pairflag stw definido en el lammps
    pairsw = new PairSW(lmp);
    char *a[6];
    a[0] = "*";
    a[1] = "*";
    a[2] = "NaCl.sw";
    a[3] = "mW";
    a[4] = "Na";
    a[5] = "Cl";
    pairsw->coeff(6,a);
    //printf("Este es el cut max %f/n",PairSW->cutmax);
    } // Matias
}

/* ---------------------------------------------------------------------- */

void FixGCMCVP::init()
{

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region {} for fix gcmc does not exist", idregion);
  }

  triclinic = domain->triclinic;

  // set probabilities for MC moves

  if (nmcmoves > 0) {
    if (pmctot == 0.0)
      if (exchmode == EXCHATOM) {  // If we're making MC moves, and haven't defined probabilities for these moves, and are exchanging atoms
        movemode = MOVEATOM;
        patomtrans = 1.0;
        pmoltrans = 0.0;
        pmolrotate = 0.0;
      } else { // If we're making MC moves, and haven't defined probabilities for these moves, and are not exchanging atoms
        movemode = MOVEMOL;
        patomtrans = 0.0;
        pmoltrans = 0.5;
        pmolrotate = 0.5;
      }
    else {  // If we're making MC moves, and we've defined probabilities for these moves
      if (pmoltrans == 0.0 && pmolrotate == 0.0)  // If we've only defined atom probabilities
        movemode = MOVEATOM;
      else
        movemode = MOVEMOL; // Otherwise, we're moving molecules
      patomtrans /= pmctot; // If we're making MC moves, and we've defined probabilities for these moves, normalize them with respect to the total probability of anything happening. 
      pmoltrans /= pmctot;
      pmolrotate /= pmctot;
    }
  } else movemode = NONE;

  // decide whether to switch to the full_energy option

  if (!full_flag) {
    if ((force->kspace) ||  // Calculate with the full energy option if we're using Kspace, eam, using a potential without a single function, using a LJ-tail-corrected potential, or a NULL potential
        (force->pair == nullptr) ||
        (force->pair->single_enable == 0) ||
        (force->pair_match("^hybrid",0)) ||
        (force->pair_match("^eam",0)) ||
        (force->pair->tail_flag)) {
      full_flag = true;  // Calculate the energy of the full system
      if (comm->me == 0)
        error->warning(FLERR,"Fix gcmc using full_energy option");
    }
  }
  
  // VP Specific -- Pair Flag management
  if (pairflag) {
    full_flag = false;
  }

  // Compute the potential energy
  if (full_flag) c_pe = modify->compute[modify->find_compute("thermo_pe")];

  int *type = atom->type;

  if (exchmode == EXCHATOM) {
    if (ngcmc_type <= 0 || ngcmc_type > atom->ntypes)
      error->all(FLERR,"Invalid atom type in fix gcmc command");
  }

  // if atoms are exchanged, warn if any deletable atom has a mol ID

  if ((exchmode == EXCHATOM) && atom->molecule_flag) {
    tagint *molecule = atom->molecule;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (type[i] == ngcmc_type)
        if (molecule[i]) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
    if (flagall && comm->me == 0)
      error->all(FLERR, "Fix gcmc cannot exchange individual atoms belonging to a molecule");
  }

  // if molecules are exchanged or moved, check for unset mol IDs

  if (exchmode == EXCHMOL || movemode == MOVEMOL) {
    tagint *molecule = atom->molecule;
    int *mask = atom->mask;
    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if (mask[i] == groupbit)
        if (molecule[i] == 0) flag = 1;
    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
    if (flagall && comm->me == 0)
      error->all(FLERR, "All mol IDs should be set for fix gcmc group atoms");
  }

  if (exchmode == EXCHMOL || movemode == MOVEMOL)
    if (atom->molecule_flag == 0 || !atom->tag_enable
        || (atom->map_style == Atom::MAP_NONE))
      error->all(FLERR, "Fix gcmc molecule command requires that atoms have molecule attributes");

  // if rigidflag defined, check for rigid/small fix
  // its molecule template must be same as this one

  fixrigid = nullptr;
  if (rigidflag) {
    int ifix = modify->find_fix(idrigid);
    if (ifix < 0) error->all(FLERR,"Fix gcmc rigid fix does not exist");
    fixrigid = modify->fix[ifix];
    int tmp;
    if (&onemols[imol] != (Molecule **) fixrigid->extract("onemol",tmp))
      error->all(FLERR, "Fix gcmc and fix rigid/small not using same molecule template ID");
  }

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one

  fixshake = nullptr;
  if (shakeflag) {
    int ifix = modify->find_fix(idshake);
    if (ifix < 0) error->all(FLERR,"Fix gcmc shake fix does not exist");
    fixshake = modify->fix[ifix];
    int tmp;
    if (&onemols[imol] != (Molecule **) fixshake->extract("onemol",tmp))
      error->all(FLERR,"Fix gcmc and fix shake not using "
                 "same molecule template ID");
  }

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix gcmc in a 2d simulation");

  // create a new group for interaction exclusions
  // used for attempted atom or molecule deletions
  // skip if already exists from previous init()

  if (full_flag && !exclusion_group_bit) {

    // create unique group name for atoms to be excluded

    auto group_id = std::string("FixGCMCVP:gcmc_exclusion_group:") + id;
    group->assign(group_id + " subtract all all");
    exclusion_group = group->find(group_id);
    if (exclusion_group == -1)
      error->all(FLERR,"Could not find fix gcmc exclusion group ID");
    exclusion_group_bit = group->bitmask[exclusion_group];

    // neighbor list exclusion setup
    // turn off interactions between group all and the exclusion group

    neighbor->modify_params(fmt::format("exclude group {} all",group_id));
  }

  // create a new group for temporary use with selected molecules

  if (exchmode == EXCHMOL || movemode == MOVEMOL) {

    // create unique group name for atoms to be rotated

    auto group_id = std::string("FixGCMCVP:rotation_gas_atoms:") + id;
    group->assign(group_id + " molecule -1");
    molecule_group = group->find(group_id);
    if (molecule_group == -1)
      error->all(FLERR,"Could not find fix gcmc rotation group ID");
    molecule_group_bit = group->bitmask[molecule_group];
    molecule_group_inversebit = molecule_group_bit ^ ~0;
  }

  // get all of the needed molecule data if exchanging
  // or moving molecules, otherwise just get the gas mass

  if (exchmode == EXCHMOL || movemode == MOVEMOL) {

    onemols[imol]->compute_mass();
    onemols[imol]->compute_com();
    gas_mass = onemols[imol]->masstotal;
    for (int i = 0; i < onemols[imol]->natoms; i++) {
      onemols[imol]->x[i][0] -= onemols[imol]->com[0];
      onemols[imol]->x[i][1] -= onemols[imol]->com[1];
      onemols[imol]->x[i][2] -= onemols[imol]->com[2];
    }
    onemols[imol]->com[0] = 0;
    onemols[imol]->com[1] = 0;
    onemols[imol]->com[2] = 0;

  } else gas_mass = atom->mass[ngcmc_type];

  if (gas_mass <= 0.0)
    error->all(FLERR,"Illegal fix gcmc gas mass <= 0");

  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do GCMC on atoms in atom_modify first group");
  }

  // compute beta, lambda, sigma, and the zz factor

  beta = 1.0/(force->boltz*reservoir_temperature);
  double lambda = sqrt(force->hplanck*force->hplanck/
                        (2.0*MY_PI*gas_mass*force->mvv2e*
                      force->boltz*reservoir_temperature));
  zz = exp(beta*chemical_potential)/(pow(lambda,3.0));


  if (!pressure_flag)    // using pressure_flag to replace pressflag defined by Matias
    zz = exp(beta * chemical_potential) / (pow(lambda, 3.0));
  else if (pressure_flag) {
    zz = pressure / (force->boltz * 4.184 * reservoir_temperature * 1000 / 6.02e23) /
        (1e30);    // from Matias; need to check the meaning
    //zz = pressure*fugacity_coeff*beta/force->nktv2p;    // from the original expression of 2015 version of lammps
  }

  // TODO line 584-596 in old gcmc_vp changes ZZ based on pressure flag. Why?
  // sigma = sqrt(force->boltz*reservoir_temperature*tfac_insert/gas_mass/force->mvv2e);
  // if (pressure_flag) zz = pressure*fugacity_coeff*beta/force->nktv2p;

  // VP Specific -- Matias' Fact 
  if (comm->me == 0) {                                     
    printf("zz factor equals %e\n", zz);                   
    printf("regionflag equals %i\n", regionflag);          
    printf("pressure_flag equals %i\n", pressure_flag);    
    printf("pairflag equals %i\n", pairflag);              
  }     
  
  imagezero = ((imageint) IMGMAX << IMG2BITS) |
             ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // warning if group id is "all"

  if ((comm->me == 0) && (groupbit & 1))
    error->warning(FLERR, "Fix gcmc is being applied "
                   "to the default group all");

  // construct group bitmask for all new atoms
  // aggregated over all group keywords

  groupbitall = 1 | groupbit;
  for (int igroup = 0; igroup < ngroups; igroup++) {
    int jgroup = group->find(groupstrings[igroup]);
    if (jgroup == -1)
      error->all(FLERR,"Could not find specified fix gcmc group ID");
    groupbitall |= group->bitmask[jgroup];
  }

  // construct group type bitmasks
  // not aggregated over all group keywords

  if (ngrouptypes > 0) {
    memory->create(grouptypebits,ngrouptypes,"fix_gcmc:grouptypebits");
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      int jgroup = group->find(grouptypestrings[igroup]);
      if (jgroup == -1)
        error->all(FLERR,"Could not find specified fix gcmc group ID");
      grouptypebits[igroup] = group->bitmask[jgroup];
    }
  }

  // current implementation is broken using
  // full_flag and translation/rotation of molecules
  // on more than one processor.

  if (full_flag && movemode == MOVEMOL && comm->nprocs > 1)
    error->all(FLERR,"fix gcmc does currently not support full_energy "
               "option with molecule MC moves on more than 1 MPI process.");

}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixGCMC::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  if (triclinic) {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  } else {
    sublo = domain->sublo;
    subhi = domain->subhi;
  }

  if (region) volume = region_volume;
  else volume = domain->xprd * domain->yprd * domain->zprd;

  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  update_gas_atoms_list();

  if (full_flag) {
    energy_stored = energy_full();
    if (overlap_flag && energy_stored > MAXENERGYTEST)
        error->warning(FLERR,"Energy of old configuration in "
                       "fix gcmc is > MAXENERGYTEST.");

    for (int i = 0; i < ncycles; i++) {
      int ixm = static_cast<int>(random_equal->uniform()*ncycles) + 1;
      if (ixm <= nmcmoves) {
        double xmcmove = random_equal->uniform();
        if (xmcmove < patomtrans) 
          attempt_atomic_translation_full();
        else if (xmcmove < patomtrans+pmoltrans)  
          error->all(FLERR, "Cannot be used for full molecule translation");
        else  
          error->all(FLERR, "Cannot be used for full molecule rotation");
      } else {
        double xgcmc = random_equal->uniform();
        if (exchmode == EXCHATOM) {
          if (xgcmc < 0.5)  
            attempt_atomic_deletion_full();
          else  
            attempt_atomic_insertion_full();
        } else {
          if (xgcmc < 0.5)  
            error->all(FLERR, "Cannot be used for full molecule deletion"); 
          else  
            error->all(FLERR, "Cannot be used for full molecule insertion"); 
        }
      }
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

  } else {
    // TODO why are only the full molecule moves forbidden?
    for (int i = 0; i < ncycles; i++) {  
      int ixm = static_cast<int>(random_equal->uniform()*ncycles) + 1;
      if (ixm <= nmcmoves) {
        double xmcmove = random_equal->uniform();
        if (xmcmove < patomtrans) 
          attempt_atomic_translation();
        else if (xmcmove < patomtrans+pmoltrans) 
          attempt_molecule_translation();
        else 
          attempt_molecule_rotation();
      } else {
        double xgcmc = random_equal->uniform();
        if (exchmode == EXCHATOM) {
          if (xgcmc < 0.5) 
            attempt_atomic_deletion();
          else 
            attempt_atomic_insertion();
        } else {
          if (xgcmc < 0.5) 
            attempt_molecule_deletion();
          else 
            attempt_molecule_insertion();
        }
      }
    }
  }
  next_reneighbor = update->ntimestep + nevery;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVP::attempt_atomic_deletion() 
{
  ndeletion_attempts += 1.0;

  if (ngas == 0 || ngas <= min_ngas) return;

  int i = pick_random_gas_atom();

  int success = 0;

  // VP Specific
  double energy_all = 0;     // Matias
  int proc_id = -2;          // Matias
  double deletion_energy;    // added by Jibao

  if (i >= 0) {
    if (atom->type[i] != ngcmc_type)
      printf("you are trying to delete an atom of type different from the one specified in fix "
             "gcmc command\natom->type[i=%d] = %d ngcmc_type = %d\n",
             i, atom->type[i], ngcmc_type);    // added by Jibao

    if (pairflag == 0) {
      deletion_energy = energy(i, ngcmc_type, -1, atom->x[i]);
    } else if (pairflag == 1) {
      pair = force->pair;    //force obtejo que tiene pair           // Matias
      deletion_energy = pairsw->Stw_GCMC(i, ngcmc_type, 1, atom->x[i]);    // Matias
    }

    energy_all = deletion_energy;    // Matias
    proc_id = comm->me;              // Matias

    //double deletion_energy = energy(i,ngcmc_type,-1,atom->x[i]);  // commented out by Jibao
    if (random_unequal->uniform() < ngas * exp(beta * deletion_energy) / (zz * volume)) {
      atom->avec->copy(atom->nlocal - 1, i, 1);
      atom->nlocal--;
      success = 1;
    }
  }

  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);

  if (success_all) {
    atom->natoms--;
    if (atom->tag_enable) {
      if (atom->map_style != Atom::MAP_NONE) atom->map_init();
    }
    atom->nghost = 0;
    if (triclinic) domain->x2lamda(atom->nlocal); 
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    update_gas_atoms_list();
    ndeletion_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVP::attempt_atomic_insertion()
{
  double lamda[3];

  ninsertion_attempts += 1.0;

  if (ngas >= max_ngas) return;

  // pick coordinates for insertion point

  double coord[3];
  if (region) {
    int region_attempt = 0;
    coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
    coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
    coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
    while (region->match(coord[0],coord[1],coord[2]) == 0) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      region_attempt++;
      if (region_attempt >= max_region_attempts) return;
    }
    if (triclinic) domain->x2lamda(coord,lamda);
  } else {
    if (triclinic == 0) {
      coord[0] = xlo + random_equal->uniform() * (xhi-xlo);
      coord[1] = ylo + random_equal->uniform() * (yhi-ylo);
      coord[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,coord);
    }
  }

  int proc_flag = 0;
  if (triclinic == 0) {
    domain->remap(coord);
    if (!domain->inside(coord))
      error->one(FLERR,"Fix gcmc put atom outside box");
    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) proc_flag = 1;
  } else {
    if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
        lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
        lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
  }

  int success = 0;

  // VP Specific
  double energy_all = 0;      // from Matias; added by Jibao
  int proc_id = -2;           // from Matias; added by Jibao
  double insertion_energy;    // added by Jibao

  if (proc_flag) {

    // VP specific
    int nall = atom->nmax;

    int ii = -1;
    if (charge_flag) {
      ii = atom->nlocal + atom->nghost;
      if (ii >= atom->nmax) atom->avec->grow(0);
      atom->q[ii] = charge;
    }

    // VP Specific
    //double insertion_energy = energy(ii,ngcmc_type,-1,coord); // commented out by Jibao

    if (!pairflag) {                                           // from Matias; added by Jibao
      insertion_energy = energy(ii,ngcmc_type,-1,coord);       // from version 2015; added by Jibao
    } else if (pairflag) {                                     // from Matias; added by Jibao
      pair = force->pair;                                      // from Matias; added by Jibao
      insertion_energy =pairsw->Stw_GCMC(nall,ngcmc_type,1,coord);  // from Matias; added by Jibao
    }
    energy_all = insertion_energy;    // from Matias; added by Jibao
    proc_id = comm->me;               // from Matias; added by Jibao
    // END VP Specific

    if (insertion_energy < MAXENERGYTEST && 
        random_unequal->uniform() < zz*volume*exp(-beta*insertion_energy)/(ngas+1)) {

      atom->avec->create_atom(ngcmc_type,coord);
      int m = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[m] = groupbitall;
      for (int igroup = 0; igroup < ngrouptypes; igroup++) {
        if (ngcmc_type == grouptypes[igroup])
          atom->mask[m] |= grouptypebits[igroup];
      }

      atom->v[m][0] = random_unequal->gaussian()*sigma;
      atom->v[m][1] = random_unequal->gaussian()*sigma;
      atom->v[m][2] = random_unequal->gaussian()*sigma;
      modify->create_attribute(m);

      success = 1;
    }
  }

  int success_all = 0;
  MPI_Allreduce(&success,&success_all,1,MPI_INT,MPI_MAX,world);

  // VP Specific
  MPI_Allreduce(&proc_id, &proc_end, 1, MPI_INT, MPI_MAX, world);    // from Matias; added by Jibao
  MPI_Bcast(&energy_all, 1, MPI_DOUBLE, proc_end, world);            // from Matias; added by Jibao
  energyout = energy_all;                                            // from Matias; added by Jibao

  if (success_all) {
    atom->natoms++;
    if (atom->tag_enable) {
      atom->tag_extend();
      if (atom->map_style != Atom::MAP_NONE) atom->map_init();
    }
    atom->nghost = 0;
    if (triclinic) domain->x2lamda(atom->nlocal);
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    update_gas_atoms_list();
    ninsertion_successes += 1.0;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVP::attempt_atomic_deletion_full()
{
  double q_tmp;
  const int q_flag = atom->q_flag;

  // VP specific
  int success = 0;
  double deletion_energy = 0.0;
  const int i = pick_random_gas_atom();
  double energy_before = 0.0;

  ndeletion_attempts += 1.0;

  if (ngas == 0 || ngas <= min_ngas) return;

  if (!pairflag) {
    energy_before = energy_stored;
  } else {
    if (i >= 0){
      pair = force->pair;
      deletion_energy = pairsw->Stw_GCMC(i, ngcmc_type, 1, atom->x[i]);
      
      // Debugging VP
      if (comm->me == 0) {
        printf("deletion_energy=pairsw->Stw_GCMC() = %f, i = %d  ",deletion_energy,i);
      }
    }
  }

  int tmpmask;
  if (i >= 0) {
    tmpmask = atom->mask[i];
    atom->mask[i] = exclusion_group_bit;
    if (q_flag) {
      q_tmp = atom->q[i];
      atom->q[i] = 0.0;
    }
  }

  if (force->kspace) force->kspace->qsum_qsq();
  if (force->pair->tail_flag) force->pair->reinit();

  // VP Specific
  double energy_after = 0.0;       // added by Jibao
  energy_after = energy_full();    // debug; Jibao

  if (!pairflag) {
    if (random_equal->uniform() <
        ngas*exp(beta*(energy_before - energy_after))/(zz*volume)) {
      if (i >= 0) {
        atom->avec->copy(atom->nlocal-1,i,1);
        atom->nlocal--;
      }
      atom->natoms--;
      if (atom->map_style != Atom::MAP_NONE) atom->map_init();
      ndeletion_successes += 1.0;
      energy_stored = energy_after;
    } else {
      if (i >= 0) {
        atom->mask[i] = tmpmask;
        if (q_flag) atom->q[i] = q_tmp;
      }
      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_stored = energy_before;
    }
  } else { // VP Specific
    double energy_all = 0;    // from Matias; added by Jibao
    int proc_id = -2;         // from Matias; added by Jibao

    energy_all = deletion_energy;    // from Matias; added by Jibao
    proc_id = comm->me;              // from Matias; added by Jibao

    if (i >= 0) {
      if (random_unequal->uniform() < ngas * exp(beta * deletion_energy) / (zz * volume)) {
        /*
                 printf("proc %d: deletion_energy = %f\n",comm->me,deletion_energy);
                 if (comm->me == 0) {
                 printf("random_unequal->uniform() < ngas*exp(beta*deletion_energy)/(zz*volume) satisfied: deletion_energy= %f, i = %d\n",deletion_energy,i);
                 }   // added by Jibao
                 */

        atom->avec->copy(atom->nlocal - 1, i, 1);
        atom->nlocal--;

        success = 1;
      }
    }

    int success_all = 0;

    int proc_end = 0;      // from Matias; added by Jibao
    MPI_Barrier(world);    // from Matias; added by Jibao
    MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);
    MPI_Allreduce(&proc_id, &proc_end, 1, MPI_INT, MPI_MAX,world);    // from Matias; added by Jibao
    MPI_Bcast(&energy_all, 1, MPI_DOUBLE, proc_end, world);    // from Matias; added by Jibao

    energyout = energy_all;    // from Matias; added by Jibao

    if (success_all) {
      atom->natoms--;

      if (atom->tag_enable) {
        if (atom->map_style) atom->map_init();
      }

      // I don't know why to set nghost to zero; need to check it!!!!!!! 
      // added and commentted by Jibao
      atom->nghost = 0;    
      comm->borders();
      ndeletion_successes += 1.0;

    } else {
      
      if (i >= 0) {
        atom->mask[i] = tmpmask;
        if (q_flag) atom->q[i] = q_tmp;
      }
      
      if (force->kspace) force->kspace->qsum_qsq();
    }
  }    // modified by Jibao
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMCVP::attempt_atomic_insertion_full()
{
  double lamda[3];
  ninsertion_attempts += 1.0;

  if (ngas >= max_ngas) return;

  double energy_before = energy_stored;

  double coord[3];
  if (region) {
    int region_attempt = 0;
    coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
    coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
    coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
    while (region->match(coord[0],coord[1],coord[2]) == 0) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      region_attempt++;
      if (region_attempt >= max_region_attempts) return;
    }
    if (triclinic) domain->x2lamda(coord,lamda);
  } else {
    if (triclinic == 0) {
      coord[0] = xlo + random_equal->uniform() * (xhi-xlo);
      coord[1] = ylo + random_equal->uniform() * (yhi-ylo);
      coord[2] = zlo + random_equal->uniform() * (zhi-zlo);
    } else {
      lamda[0] = random_equal->uniform();
      lamda[1] = random_equal->uniform();
      lamda[2] = random_equal->uniform();

      // wasteful, but necessary

      if (lamda[0] == 1.0) lamda[0] = 0.0;
      if (lamda[1] == 1.0) lamda[1] = 0.0;
      if (lamda[2] == 1.0) lamda[2] = 0.0;

      domain->lamda2x(lamda,coord);
    }
  }

  int proc_flag = 0;
  if (triclinic == 0) {
    domain->remap(coord);
    if (!domain->inside(coord))
      error->one(FLERR,"Fix gcmc put atom outside box");
    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) proc_flag = 1;
  } else {
    if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
        lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
        lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
  }

  if (proc_flag) {
    atom->avec->create_atom(ngcmc_type,coord);
    int m = atom->nlocal - 1;

    // add to groups
    // optionally add to type-based groups

    atom->mask[m] = groupbitall;
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      if (ngcmc_type == grouptypes[igroup])
        atom->mask[m] |= grouptypebits[igroup];
    }

    atom->v[m][0] = random_unequal->gaussian()*sigma;
    atom->v[m][1] = random_unequal->gaussian()*sigma;
    atom->v[m][2] = random_unequal->gaussian()*sigma;
    if (charge_flag) atom->q[m] = charge;
    modify->create_attribute(m);
  }

  atom->natoms++;
  if (atom->tag_enable) {
    atom->tag_extend();
    if (atom->map_style != Atom::MAP_NONE) atom->map_init();
  }
  atom->nghost = 0;
  if (triclinic) domain->x2lamda(atom->nlocal);
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (force->kspace) force->kspace->qsum_qsq();
  if (force->pair->tail_flag) force->pair->reinit();
  double energy_after = energy_full();

  if (energy_after < MAXENERGYTEST &&
      random_equal->uniform() <
      zz*volume*exp(beta*(energy_before - energy_after))/(ngas+1)) {

    ninsertion_successes += 1.0;
    energy_stored = energy_after;
  } else {
    atom->natoms--;
    if (proc_flag) atom->nlocal--;
    if (force->kspace) force->kspace->qsum_qsq();
    if (force->pair->tail_flag) force->pair->reinit();
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
  return acceptance ratios
------------------------------------------------------------------------- */

double FixGCMCVP::compute_vector(int n)
{
  if (n == 0) return ntranslation_attempts;
  if (n == 1) return ntranslation_successes;
  if (n == 2) return ninsertion_attempts;
  if (n == 3) return ninsertion_successes;
  if (n == 4) return ndeletion_attempts;
  if (n == 5) return ndeletion_successes;
  if (n == 6) return nrotation_attempts;
  if (n == 7) return nrotation_successes;
  if (n == 8) return energy_out;
  return 0.0;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixGCMC::update_gas_atoms_list()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double **x = atom->x;

  if (atom->nmax > gcmc_nmax) {
    memory->sfree(local_gas_list);
    gcmc_nmax = atom->nmax;
    local_gas_list = (int *) memory->smalloc(gcmc_nmax*sizeof(int),
     "GCMC:local_gas_list");
  }

  ngas_local = 0;

  // VP Specific
  int *type = atom->type;

  if (region) {

    if (exchmode == EXCHMOL || movemode == MOVEMOL) {

      tagint maxmol = 0;
      for (int i = 0; i < nlocal; i++) maxmol = MAX(maxmol,molecule[i]);
      tagint maxmol_all;
      MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
      auto comx = new double[maxmol_all];
      auto comy = new double[maxmol_all];
      auto comz = new double[maxmol_all];
      for (int imolecule = 0; imolecule < maxmol_all; imolecule++) {
        for (int i = 0; i < nlocal; i++) {
          if (molecule[i] == imolecule) {
            mask[i] |= molecule_group_bit;
          } else {
            mask[i] &= molecule_group_inversebit;
          }
        }
        double com[3];
        com[0] = com[1] = com[2] = 0.0;
        group->xcm(molecule_group,gas_mass,com);

        // remap unwrapped com into periodic box

        domain->remap(com);
        comx[imolecule] = com[0];
        comy[imolecule] = com[1];
        comz[imolecule] = com[2];
      }

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          if (region->match(comx[molecule[i]],
             comy[molecule[i]],comz[molecule[i]]) == 1) {
            local_gas_list[ngas_local] = i;
            ngas_local++;
          }
        }
      }
      delete[] comx;
      delete[] comy;
      delete[] comz;
    } else {
      for (int i = 0; i < nlocal; i++) {
        if ((mask[i] & groupbit) && (type[i] == ngcmc_type)) {  // VP Specific
          if (region->match(x[i][0],x[i][1],x[i][2]) == 1) {
            local_gas_list[ngas_local] = i;
            ngas_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && (type[i] == ngcmc_type)) {  // VP Specific
        local_gas_list[ngas_local] = i;
        ngas_local++;
      }
    }
  }

  MPI_Allreduce(&ngas_local,&ngas,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&ngas_local,&ngas_before,1,MPI_INT,MPI_SUM,world);
  ngas_before -= ngas_local;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGCMC::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal fix gcmc command");

  // defaults

  exchmode = EXCHATOM;
  movemode = NONE;
  patomtrans = 0.0;
  pmoltrans = 0.0;
  pmolrotate = 0.0;
  pmctot = 0.0;
  max_rotation_angle = 10*MY_PI/180;
  region_volume = 0;
  max_region_attempts = 1000;
  molecule_group = 0;
  molecule_group_bit = 0;
  molecule_group_inversebit = 0;
  exclusion_group = 0;
  exclusion_group_bit = 0;
  pressure_flag = false;
  pressure = 0.0;
  fugacity_coeff = 1.0;
  rigidflag = 0;
  shakeflag = 0;
  charge = 0.0;
  charge_flag = false;
  full_flag = false;
  ngroups = 0;
  int ngroupsmax = 0;
  groupstrings = nullptr;
  ngrouptypes = 0;
  int ngrouptypesmax = 0;
  grouptypestrings = nullptr;
  grouptypes = nullptr;
  grouptypebits = nullptr;
  energy_intra = 0.0;
  tfac_insert = 1.0;
  overlap_cutoffsq = 0.0;
  overlap_flag = 0;
  min_ngas = -1;
  max_ngas = INT_MAX;

  int iarg = 0;
  while (iarg < narg) {
  if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1)
        error->all(FLERR,"Molecule template ID for fix gcmc does not exist");
      if (atom->molecules[imol]->nset > 1 && comm->me == 0)
        error->warning(FLERR,"Molecule template for "
                       "fix gcmc has multiple molecules");
      exchmode = EXCHMOL;
      onemols = atom->molecules;
      nmol = onemols[imol]->nset;
      iarg += 2;
  } else if (strcmp(arg[iarg],"mcmoves") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix gcmc command");
      patomtrans = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      pmoltrans = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      pmolrotate = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      if (patomtrans < 0 || pmoltrans < 0 || pmolrotate < 0)
        error->all(FLERR,"Illegal fix gcmc command");
      pmctot = patomtrans + pmoltrans + pmolrotate;
      if (pmctot <= 0)
        error->all(FLERR,"Illegal fix gcmc command");
      iarg += 4;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      region = domain->get_region_by_id(arg[iarg+1]);
      if (!region)
        error->all(FLERR,"Region {} for fix gcmc does not exist",arg[iarg+1]);
      idregion = utils::strdup(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"maxangle") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      max_rotation_angle = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      max_rotation_angle *= MY_PI/180;
      iarg += 2;

    // VP Specific -- Matias
    } else if (strcmp(arg[iarg], "pair") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix GCMC command");
      if (strcmp(arg[iarg + 1], "lj/cut") == 0)
        pairflag = 0;
      else if (strcmp(arg[iarg + 1], "Stw") == 0)
        pairflag = 1;
      else
        error->all(FLERR, "Illegal fix evaporate command");
      iarg += 2;
    }    
    
    } else if (strcmp(arg[iarg],"pressure") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      pressure = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      pressure = pressure * 100.0;    // VP Specifc added by Jibao, according to Matias' code
      pressure_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"fugacity_coeff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      fugacity_coeff = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"charge") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      charge = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      charge_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"rigid") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      delete [] idrigid;
      idrigid = utils::strdup(arg[iarg+1]);
      rigidflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"shake") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      delete [] idshake;
      idshake = utils::strdup(arg[iarg+1]);
      shakeflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"full_energy") == 0) {
      full_flag = true;
      iarg += 1;
    } else if (strcmp(arg[iarg],"group") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      if (ngroups >= ngroupsmax) {
        ngroupsmax = ngroups+1;
        groupstrings = (char **)
          memory->srealloc(groupstrings,
                           ngroupsmax*sizeof(char *),
                           "fix_gcmc:groupstrings");
      }
      groupstrings[ngroups] = utils::strdup(arg[iarg+1]);
      ngroups++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"grouptype") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix gcmc command");
      if (ngrouptypes >= ngrouptypesmax) {
        ngrouptypesmax = ngrouptypes+1;
        grouptypes = (int*) memory->srealloc(grouptypes,ngrouptypesmax*sizeof(int),
                         "fix_gcmc:grouptypes");
        grouptypestrings = (char**)
          memory->srealloc(grouptypestrings,
                           ngrouptypesmax*sizeof(char *),
                           "fix_gcmc:grouptypestrings");
      }
      grouptypes[ngrouptypes] = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      grouptypestrings[ngrouptypes] = utils::strdup(arg[iarg+2]);
      ngrouptypes++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"intra_energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      energy_intra = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tfac_insert") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      tfac_insert = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"overlap_cutoff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      double rtmp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      overlap_cutoffsq = rtmp*rtmp;
      overlap_flag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"min") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      min_ngas = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"max") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix gcmc command");
      max_ngas = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix gcmc command");
  }
}


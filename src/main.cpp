
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <cassert>

#include "GenericIO.h"

#define POSVEL_T float
#define ID_T int64_t
#define MASK_T uint16_t

struct pos_t {
  POSVEL_T x, y, z, w;
};

using namespace std;
using namespace gio;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int Rank, NRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &NRanks);

  bool UseAOS = false;
  int a = 1;
  
  if(argc != 2) {
    fprintf(stderr,"USAGE: %s <mpiioName>\n", argv[0]);
    exit(-1);
  }

  char *mpiioName = argv[a];
  
  GenericIO::setNaturalDefaultPartition();

  vector<POSVEL_T> xx, yy, zz, vx, vy, vz, phi;
  vector<ID_T> id;
  vector<MASK_T> mask;

  vector<pos_t> pos, vel;

  assert(sizeof(ID_T) == 8);

  size_t Np = 0;
  unsigned Method = GenericIO::FileIOPOSIX;
  const char *EnvStr = getenv("GENERICIO_USE_MPIIO");
  if (EnvStr && string(EnvStr) == "1")
    Method = GenericIO::FileIOMPI;

  { // scope GIO
    GenericIO GIO(
        MPI_COMM_WORLD,
        mpiioName, Method);
    GIO.openAndReadHeader(GenericIO::MismatchRedistribute);
    
    int NR = GIO.readNRanks();
        if (!Rank && NR != NRanks) {
            cout << "Redistributing data from " << NR << " ranks to " << NRanks <<
                    " ranks; dropping rank topology information!\n";
        }

    MPI_Barrier(MPI_COMM_WORLD);

    Np = GIO.readNumElems();
    
    double PhysOrigin[3], PhysScale[3];
    GIO.readPhysOrigin(PhysOrigin);
    GIO.readPhysScale(PhysScale);
    
    cout << "PhysOrigin " << PhysOrigin << " ; PhysScale " << PhysScale << endl;
    cout << "Parts " << Np << endl;
    
    xx.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    yy.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    zz.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    vx.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    vy.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    vz.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    phi.resize(Np + GIO.requestedExtraSpace()/sizeof(POSVEL_T));
    id.resize(Np + GIO.requestedExtraSpace()/sizeof(ID_T));
    mask.resize(Np + GIO.requestedExtraSpace()/sizeof(MASK_T));

    GIO.addVariable("x", xx, GenericIO::VarHasExtraSpace);
    GIO.addVariable("y", yy, GenericIO::VarHasExtraSpace);
    GIO.addVariable("z", zz, GenericIO::VarHasExtraSpace);
    GIO.addVariable("vx", vx, GenericIO::VarHasExtraSpace);
    GIO.addVariable("vy", vy, GenericIO::VarHasExtraSpace);
    GIO.addVariable("vz", vz, GenericIO::VarHasExtraSpace);
    GIO.addVariable("phi", phi, GenericIO::VarHasExtraSpace);
    GIO.addVariable("id", id, GenericIO::VarHasExtraSpace);
    GIO.addVariable("mask", mask, GenericIO::VarHasExtraSpace);

    GIO.readData();
  } // destroy GIO prior to calling MPI_Finalize

  xx.resize(Np);
  yy.resize(Np);
  zz.resize(Np);
  vx.resize(Np);
  vy.resize(Np);
  vz.resize(Np);
  phi.resize(Np);
  id.resize(Np);
  mask.resize(Np);
  
  

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}


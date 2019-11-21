import sys
import os
from mpi4py import MPI
import generic_io
import numpy as np
from pmesh.pm import ParticleMesh
from absl import app
from absl import flags
from astropy.io import fits

FLAGS = flags.FLAGS

flags.DEFINE_string("snapshot",
                    "/global/projecta/projectdirs/lsst/groups/CS/cosmoDC2/Outer_snapshots/z1.01/m000.mpicosmo.247",
                    "Path to the GenericIO snapshot file.")

flags.DEFINE_string("galaxy_positions",
                    "/global/cscratch1/sd/flanusse/pos_247.fits",
                    "Path to galaxy positions")

flags.DEFINE_string("output_dir",
                    "/global/cscratch1/sd/flanusse",
                    "Output directory to save the  tidal field information.")

flags.DEFINE_integer("mesh_size", 2048,
                     "Size of the mesh on which the tidal field is computed.")

flags.DEFINE_float("smoothing_scale", 1.0,
                   "Smoothing scale for Gaussian low pass filter, in Mpc/h")

# Create communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

def lowpass_transfer(r):
  """
  Filter for smoothing the field
  """
  def filter(k, v):
    k2 = sum(ki ** 2 for ki in k)
    return np.exp(-0.5 * k2 * r**2) * v
  return filter

def tidal_transfer(d1, d2):
  """
  Filter to compute the tidal tensor
  """
  def filter(k, v):
    k2 = sum(ki ** 2 for ki in k)
    k2[k2 == 0] = 1.0
    C1 = (v.BoxSize / v.Nmesh)[d1]
    w1 = k[d1] * C1

    C2 = (v.BoxSize / v.Nmesh)[d2]
    w2 = k[d2] * C2
    return w1 * w2 / k2 * v
  return filter

def main(argv):

  # Read the particle data
  gio = generic_io.Generic_IO(FLAGS.snapshot, comm)
  metadata = gio.read_metadata(rank=rank)
  if rank == 0:
    print("Snapshot metadata", metadata)

  # Create the mesh
  pm =  ParticleMesh(BoxSize=metadata['scale'][0], # assuming cube
                     Nmesh=[FLAGS.mesh_size]*3,
                     dtype='f8',
                     comm=comm)
  # Block size for reading the data
  block_size = metadata['num_ranks'] // comm.Get_size()
  pos = gio.read_columns(["x", "y", "z"],
                         as_numpy_array=True,
                         ranks=list(range(rank, metadata['num_ranks'], nranks)))
  pos = np.asarray([list(sublist) for sublist in pos])

  # Create domain decomposition for the particles that matches the mesh
  # decomposition
  layout = pm.decompose(pos)

  # Create a mesh
  rho = pm.create('real')

  # Paint the particles on the mesh
  rho.paint(pos, layout=layout, hold=False)
  print('Density painted')

  # Compute density and forward FFT
  N = pm.comm.allreduce(len(pos))
  fac = 1.0 * pm.Nmesh.prod() / N
  rho[...] *= fac
  rhok = rho.r2c()
  rhok = rhok.apply(lowpass_transfer(r=FLAGS.smoothing_scale))
  print('Rho computed')

  # Density field is loaded, now retrieving the galaxy positions
  if rank==0:
    gal_pos = fits.getdata(FLAGS.galaxy_positions)
  else:
    gal_pos = np.array([[],[],[]]).reshape([-1,3])

  # Computing the distribution of galaxies in the cube
  layout_gal = pm.decompose(gal_pos)
  gal_pos = layout_gal.exchange(gal_pos)

  # Retrieve local density on each galaxy
  density = rhok.c2r().readout(gal_pos)
  density = layout_gal.gather(density, mode='all')
  if rank == 0:
    fits.writeto(FLAGS.output_dir+'/density_247.fits', density, overwrite=True)

  tidal_tensors = []
  for i in range(3):
    tidal_tensors.append(np.stack([rhok.apply(tidal_transfer(j, i)).c2r().readout(gal_pos) for j in range(3)], axis=-1))
  tidal_tensors = np.stack(tidal_tensors, axis=-1)

  # At this point, tidal_tensor in each rank contains the tensor for the local
  # galaxies
  # Now computing diagonalization
  vals, vects = np.linalg.eigh(tidal_tensors)

  # Retrieving the computed values
  vals = layout_gal.gather(vals, mode='all')
  vects = layout_gal.gather(vects, mode='all')

  if rank == 0:
    fits.writeto(FLAGS.output_dir+'/tidal_val_247.fits', vals, overwrite=True)
    fits.writeto(FLAGS.output_dir+'/tidal_vects_247.fits', vects, overwrite=True)


if __name__ == "__main__":
  app.run(main)

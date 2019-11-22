import numpy as np, halotools

class Randomalignment(object):

      def __init__(self, gal_type):

          self.gal_type = gal_type
          self._mock_generation_calling_sequence = ['assign_randomellipsoid','project_ellipsoid']
          self._galprop_dtypes_to_allocate = np.dtype([('gal_ellipsoid',('f4',3,3)),(gal_axes,('f4',2)),(gal_ellip,('f4',2))])
          self.list_of_haloprops_needed = ['halo_mass']

     def assign_randomellipsoid(self,**kwargs):
         table = kwargs['table']
        rot_ident = np.identity(3)
        rand_alphas = np.random.rand(self.ngal)*2.0*np.pi
        rand_betas = np.random.rand(self.ngal)*1.0*np.pi
        rand_gammas = np.random.rand(ngal)*2.0*np.pi
        rand_qs = (np.random.rand(ngal*2)).reshape((ngal,2))
        rand_q = np.max(rand_qs,axis=1)
        rand_s = np.min(rand_qs,axis=1)
        phic = np.cos(rand_alphas)
        phis = np.sin(rand_alphas)
        thetac = np.cos(rand_betas)
        thetas = np.sin(rand_betas)
        psic = np.cos(rand_gammas)
        psis = np.sin(rand_gammas)

        ellipsoid[:,0,0] = psic*phic - thetac*phis*psis
        ellipsoid[:,0,1] = psic*phis + thetac*phic*psis
        ellipsoid[:,0,2] = psis*thetas
        ellipsoid[:,1,0] = -psis*phic - thetac*phis*psic
        ellipsoid[:,1,1] = -psis*phis + thetac*phic*psic
        ellipsoid[:,1,2] = psic*thetas
        ellipsoid[:,2,0] = thetas*phis
        ellipsoid[:,2,1] = -thetas*phic
        ellipsoid[:,2,2] = thetac


     def project_ellipsoid(self,**kwargs):
         table = kwargs['table']

        s1 = eigvecs[0]
        s2 = eigvecs[1]
        s3 = eigvecs[2]

        wu = np.array([1,qsvals[0]**2,qsvals[1]**2])

        wu1 = wu[0]
        wu2 = wu[1]
        wu3 = wu[2]

        sperp1 = np.matrix([[s1[0]],[s1[1]]])
        sperp2 = np.matrix([[s2[0]],[s2[1]]])
        sperp3 = np.matrix([[s3[0]],[s3[1]]])

        kappa = s1[2]*sperp1/wu1**2 + s2[2]*sperp2/wu2**2 + s3[2]*sperp3/wu3**2
        alpha2 = (s1[2]/wu1)**2.0 + (s2[2]/wu2)**2.0 + (s3[2]/wu3)**2.0
        winv = np.matmul(sperp1,sperp1.T)/wu1**2 + np.matmul(sperp2,sperp2.T)/wu2**2 + np.matmul(sperp3,sperp3.T)/wu3**2 - np.matmul(kappa,kappa.T)/alpha2
        Wmat = np.linalg.inv(winv)
        detW = np.linalg.det(Wmat)

        ellip1 = (Wmat[0][0] - Wmat[1][1])/(Wmat[0][0] + Wmat[1][1] + 2.0*np.sqrt(detW))
        ellip2 = (2.0*Wmat[0][1])/(Wmat[0][0] + Wmat[1][1] + 2.0*np.sqrt(detW))

from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import HodModelFactory
redshift = data['redshift']
Lbox = 5000.0
particle_mass = 2.6e9
num_halos = len(redshift)
halo_catalog = UserSuppliedHaloCatalog(redshift=redshift,Lbox=Lbox,particle_mass=particle_mass,halo_x=data['x'],halo_y=data['y'],halo_z=-data['z'],halo_id=data['id'])
cen_align = Randomalignment(Centrals)
new_model = HodModelFactory()
new_model.populate_mock(halo_catalog)

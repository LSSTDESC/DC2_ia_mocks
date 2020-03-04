import numpy as np, halotools
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import SubhaloModelFactory

import GCRCatalogs
import pyccl as ccl

class Randomalignment(object):

      def __init__(self):

          self._mock_generation_calling_sequence = ['assign_randomellipsoid','project_ellipsoid']
          self._galprop_dtypes_to_allocate = np.dtype([('gal_ellipsoid00','f4'),('gal_ellipsoid01','f4'),('gal_ellipsoid02','f4'),('gal_ellipsoid10','f4'),('gal_ellipsoid11','f4'),('gal_ellipsoid12','f4'),('gal_ellipsoid20','f4'),('gal_ellipsoid21','f4'),('gal_ellipsoid22','f4'),('gal_axesratio_q','f4'),('gal_axesratio_s','f4'),('gal_ellip1','f4'),('gal_ellip2','f4')])
          self.list_of_haloprops_needed = ['halo_mass']

      def assign_randomellipsoid(self,**kwargs):

          table = kwargs['table']

          hmass = table['halo_mass']
          ngal = len(hmass)
          print(ngal)

          rand_alphas = np.random.rand(ngal)*2.0*np.pi
          rand_betas = np.random.rand(ngal)*1.0*np.pi
          rand_gammas = np.random.rand(ngal)*2.0*np.pi

          phic = np.cos(rand_alphas)
          phis = np.sin(rand_alphas)
          thetac = np.cos(rand_betas)
          thetas = np.sin(rand_betas)
          psic = np.cos(rand_gammas)
          psis = np.sin(rand_gammas)

          table['gal_ellipsoid00']= psic*phic - thetac*phis*psis
          table['gal_ellipsoid01'] = psic*phis + thetac*phic*psis
          table['gal_ellipsoid02'] = psis*thetas
          table['gal_ellipsoid10'] = -psis*phic - thetac*phis*psic
          table['gal_ellipsoid11'] = -psis*phis + thetac*phic*psic
          table['gal_ellipsoid12'] = psic*thetas
          table['gal_ellipsoid20'] = thetas*phis
          table['gal_ellipsoid21'] = -thetas*phic
          table['gal_ellipsoid22'] = thetac

      def func_proj(self,s1,s2,s3):

          rand_qs = np.random.rand(2)
          rand_q = np.max(rand_qs)
          rand_s = np.min(rand_qs)

          wu = np.array([1,rand_q**2,rand_s**2])

          wu1 = wu[0]
          wu2 = wu[1]
          wu3 = wu[2]

          sperp1 = np.matrix([[s1[0]],[s1[1]]])
          sperp2 = np.matrix([[s2[0]],[s2[1]]])
          sperp3 = np.matrix([[s3[0]],[s3[1]]])

          kappa = s1[2]*sperp1/wu1**2 + s2[2]*sperp2/wu2**2 + s3[2]*sperp3/wu3**2
          alpha2 = (s1[2]/wu1)**2.0 + (s2[2]/wu2)**2.0 + (s3[2]/wu3)**2.0
          winv = np.matmul(sperp1,sperp1.T)/wu1**2 + np.matmul(sperp2,sperp2.T)/wu2**2 + np.matmul(sperp3,sperp3.T)/wu3**2 - np.matmul(kappa,kappa.T)/alpha2

          try:
             Wmat = np.linalg.inv(winv)
             detW = np.linalg.det(Wmat)
             ellip1 = (Wmat[0,0] - Wmat[1,1])/(Wmat[0,0] + Wmat[1,1] + 2.0*np.sqrt(detW))
             ellip2 = (2.0*Wmat[0,1])/(Wmat[0,0] + Wmat[1,1] + 2.0*np.sqrt(detW))
             rand_qi = rand_q
             rand_si = rand_s
             ellip1i = ellip1
             ellip2i = ellip2
          except:
             rand_qi = 0.0
             rand_si = 0.0
             ellip1i = 0.0
             ellip2i = 0.0

          return rand_qi, rand_si, ellip1i, ellip2i

      def project_ellipsoid(self,**kwargs):

          table = kwargs['table']
          hmass = table['halo_mass']
          ngal = len(hmass)
          frand_q = np.zeros(ngal,dtype='f4')
          frand_s = np.zeros(ngal,dtype='f4')
          fellip1 = np.zeros(ngal,dtype='f4')
          fellip2 = np.zeros(ngal,dtype='f4')
          for i in range(0,ngal):
                sge1 = np.array([table['gal_ellipsoid00'][i],table['gal_ellipsoid01'][i],table['gal_ellipsoid02'][i]])
                sge2 = np.array([table['gal_ellipsoid10'][i],table['gal_ellipsoid11'][i],table['gal_ellipsoid12'][i]])
                sge3 = np.array([table['gal_ellipsoid20'][i],table['gal_ellipsoid21'][i],table['gal_ellipsoid22'][i]])
                frand_q[i],frand_s[i],fellip1[i],fellip2[i] = self.func_proj(sge1,sge2,sge3)
                #print(i,ngal)
          table['gal_axesratio_q'] = frand_q
          table['gal_axesratio_s'] = frand_s
          table['gal_ellip1'] = fellip1
          table['gal_ellip2'] = fellip2 

# Test: Read GCR catalog
gc = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_small')
data = gc.get_quantities(['halo_mass', 'redshift', 'position_x', 'position_y', 'position_z', 'is_central', 'halo_id','velocity_x','velocity_y','velocity_z','galaxy_id','R_v'], filters=['redshift <= 1.001', 'redshift >= 0.999'])

Lbox = 4225.0
partmass = 2.6e9
redshift_fake=0.0

# Create halotools like halocatalog
halo_catalog = UserSuppliedHaloCatalog(particle_mass=partmass,redshift=redshift_fake,Lbox=Lbox,halo_redshift=data['redshift'],halo_iscentral=data['is_central'],halo_x=data['position_x'],halo_y=data['position_y'],halo_z=data['position_z']+Lbox,halo_id=data['galaxy_id'],halo_mvir=data['halo_mass'],halo_vx=data['velocity_x'],halo_vy=data['velocity_y'],halo_vz=data['velocity_z'],halo_upid=data['halo_id'],halo_mass=data['halo_mass'],halo_hostid=data['halo_id'],halo_rvir=pow(data['halo_mass'],1.0/3.0))

# Random alignment + 2D shapes
galrand_align = Randomalignment()
new_model = SubhaloModelFactory(random_gals=galrand_align)

# Populate mock
new_model.populate_mock(halo_catalog)

# Halocatalog table with mock shapes
fmock_table = new_model.mock.galaxy_table

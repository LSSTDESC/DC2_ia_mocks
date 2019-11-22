import numpy as np, GCRCatalogs


class IA_gcr(object):

    def __init__(self, catname, catalogs=GCRCatalogs):

        self.catname = 'cosmoDC2_v1.1.4_small'
        self.redshift = '1.0'
        self.massthresh = '1e13'
        self.gc = catalogs.load_catalog(self.catname)
        self.data = gc.get_quantities(['halo_mass', 'stellar_mass','position_x', 'position_y', 'position_z','is_central','halo_id','redshift'], filters=['redshift <= ' + str(self.redshift + 0.001), 'redshift >= ' + str(self.redshift - 0.001)])
        self.iscentral = self.data['iscentral']
        self.ngal = len(self.iscentral)
        self.ellipsoid = np.zeros((ngal,3,3),dtype=np.float64)  
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

    def proj_ellipsoid(eigvecs,qsvals):

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



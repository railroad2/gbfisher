import os
import sys
import datetime

import numpy as np
import healpy as hp
import camb


k = 1.38064852e-23 # m^2 kg s^-2 K^-1
c = 2.99792458e+08 # m/s

args_cosmology = ['H0', 'cosmomc_theta', 'ombh2', 'omch2', 'omk', 
                  'neutrino_hierarchy', 'num_massive_neutrinos',
                  'mnu', 'nnu', 'YHe', 'meffsterile', 'standard_neutrino_neff', 
                  'TCMB', 'tau', 'deltazrei', 'bbnpredictor', 'theta_H0_range'] 

args_InitPower = ['As', 'ns', 'nrun', 'nrunrun', 'r', 'nt', 'ntrun', 'pivot_scalar', 
                  'pivot_tensor', 'parameterization']

def dl2cl(dls):
    """ Convert the angular spectrum D_l to C_l.
    C_l = D_l * 2 * np.pi / l / (l+1)
    
    Parameters
    ----------
    dls : array
        Angular spectrum, D_l, to be converted. 

    Returns
    -------
    cls : array
        Converted array.
    """
    if (arr_rank(dls)==1):
        cls = dls.copy()
        ell = np.arange(len(cls))
        cls[1:] = cls[1:] * (2. * np.pi) / (ell[1:] * (ell[1:] + 1))
    elif (arr_rank(dls)==2):
        if (len(dls) < 10):
            cls = dls.copy()
            ell = np.arange(len(cls[0]))
            for i in range(len(cls)):
                cls[i][1:] = cls[i][1:] * (2. * np.pi) \
                             / (ell[1:] * (ell[1:]+1))
        else:
            cls = np.transpose(dls.copy())
            ell = np.arange(len(cls[0]))
            for i in range(len(cls)):
                cls[i][1:] = cls[i][1:] * (2. * np.pi) \
                             / (ell[1:] * (ell[1:]+1))
            cls = np.transpose(cls) 
    return cls


def cl2dl(cls):
    """ Convert the angular spectrum C_l to D_l.
    D_l = C_l * l * (l+1) / 2 / pi
    
    Parameters
    ----------
    cls : array
        Angular spectrum, C_l, to be converted. 

    Returns
    -------
    dls : array
        Converted array.
    """
    if (arr_rank(cls)==1):
        dls = cls.copy()
        ell = np.arange(len(dls))
        dls[1:] = dls[1:] / (2. * np.pi) * (ell[1:] * (ell[1:] + 1))
    elif (arr_rank(cls)==2):
        if (len(cls) < 10):
            dls = cls.copy()
            ell = np.arange(len(dls[0]))
            for i in range(len(dls)):
                dls[i][1:] = dls[i][1:] / (2. * np.pi) \
                             * (ell[1:] * (ell[1:]+1))
        else:
            dls = np.transpose(cls.copy())
            ell = np.arange(len(dls[0]))
            for i in range(len(dls)):
                dls[i][1:] = dls[i][1:] / (2. * np.pi) \
                             * (ell[1:] * (ell[1:]+1))
            dls = np.transpose(dls) 
    else:
        print('cls has shape of', cls.shape)
        return 

    return dls


def arg2dict(**kwargs):
    ## arguments to dictionaries
    kwargs_cosmology={}
    kwargs_InitPower={}

    for key, value in kwargs.items():  # for Python 3, items() instead of iteritems()
        if key in args_cosmology: 
            kwargs_cosmology[key]=value
        elif key in args_InitPower:
            kwargs_InitPower[key]=value
        else:
            print('Wrong keyword: ' + key)

    return kwargs_cosmology, kwargs_InitPower


def get_spectrum_camb(lmax, 
                      isDl=False, cambres=False, TTonly=False, unlensed=False, CMB_unit='muK', 
                      inifile=None,
                      **kwargs):
    """
    """
   
    kwargs_cosmology, kwargs_InitPower = arg2dict(**kwargs)


    ## call camb
    if inifile is None:
        pars = camb.CAMBparams()
    else:
        pars = camb.read_ini(inifile)

    kwargs_cosmology['H0'] = pars.H0
    pars.set_cosmology(**kwargs_cosmology)
    pars.InitPower.set_params(**kwargs_InitPower)
    pars.WantTensors = True

    results = camb.get_results(pars)

    raw_cl = np.logical_not(isDl)

    if unlensed:
        cls = results.get_unlensed_total_cls(lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl).T
    else:
        cls = results.get_total_cls(lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl).T

    if TTonly:
        cls = cls[0]

    if cambres:
        return cls, results
    else:
        return cls 
    

def get_spectrum_noise(lmax, wp, fwhm=None, isDl=False, TTonly=False, CMB_unit='muK'):
    cls = np.array([(np.pi/10800 * wp * 1e-6) ** 2]*(lmax+1)) # wp is w_p^(-0.5) in uK arcmin
    cls[0] = cls[1] = 0

    if (CMB_unit == 'muK'):
        cls *= 1e12

    if fwhm:
        ell = np.arange(lmax+1)
        cls *= np.exp(ell**2 * fwhm * (np.pi/180)**2 / 8 / np.log(2))

    if (not TTonly):
        cls = np.array([cls, cls*2, cls*2, cls*0]) #+ [np.zeros(cls.shape)])

    if (isDl):
        res = cl2dl(cls)
    else:
        res = cls

    return res


def variance_cl(cls):
    """ Variance of angular spectrum C_l

    Parameters
    ----------
    cls : array
        Angular power spectrum.

    Returns
    -------
    var : float
        variance of the angular power spectrum.
    """
    if (arr_rank(cls)==1):
        ell = np.arange(len(cls))
        var = np.sum((2*ell+1)*cls/4./np.pi) 
    else:
        var = []
        ell = np.arange(len(cls[0]))
        for i in range(len(cls)):
            var.append(np.sum((2*ell+1)*cls[i]/4./np.pi))

    return var


def today():
    """ Returns the date of today in string."""
    return datetime.datetime.now().strftime('%Y-%m-%d')


def binmask(m, cut=0.8):
    m1 = m.copy()
    m1[m1>=cut] = 1
    m1[m1<cut] = 0

    return m1


def makegbmask(nside, low, high):
    mask = np.zeros(12*nside**2)
    npix = hp.query_strip(nside, np.radians(low), np.radians(high))
    mask[npix] = 1
    return mask
    

def gbmask_cut_edge(map_in, angle, nside_out=None, deg=True):
    nside_in = hp.npix2nside(len(map_in))
    if nside_out is None:
        nside_out = nside_in

    npix_out = hp.nside2npix(nside_out)

    map_tmp = map_in.copy()
    map_tmp[map_in == 0] = hp.UNSEEN
    idx = np.arange(len(map_tmp))
    idx_obs = idx[map_tmp != hp.UNSEEN]
    theta0, phi0 = hp.pix2ang(nside_in, idx_obs[0])
    theta1, phi1 = hp.pix2ang(nside_in, idx_obs[-1])

    if deg:
        angle = np.radians(angle)

    idx_strip = hp.query_strip(nside_out, theta0+angle, theta1-angle)

    map_out = np.full(npix_out, 0) #hp.UNSEEN) 
    map_out[idx_strip] = 1 

    return map_out
                            

def mymask(maskfname=None, coord='equ', nside=8, 
           nomask=False, gbmask=True, galmask=True, syncmask=False, bwmask=True):

    nside0 = 1024
    if maskfname is None:
        saveflg = False
    else:
        try:
            mask = hp.read_map(maskfname)
            return np.array(mask)
        except:
            print (f'The file {maskfname} does not exist. Creating the mask.')
            saveflg = True

    ## gb mask
    tilt = 20
    latOT = 61.7 
    mmin = latOT - tilt - 10
    mmax = latOT + tilt + 10 
    if gbmask:
        mask_gb = makegbmask(nside0, mmin, mmax)
        rot = hp.rotator.Rotator(coord=['c','g'])
    else:
        mask_gb = np.ones(12*nside0*nside0)

    ## galactic mask
    if galmask:
        fn_galmask = f'./maps/mask_gal_ns1024_equ.fits'
        mask_gal = hp.read_map(fn_galmask, verbose=0, dtype=None)
    else:
        mask_gal = np.ones(12*nside0*nside0) 

    ## synchrotron mask
    if syncmask:
        fn_syncmask = f'./maps/mask_sync_ns1024_equ.fits'
        mask_sync = hp.read_map(fn_syncmask, verbose=0, dtype=None)
    else:
        mask_sync = np.ones(12*nside0*nside0)

    mask = mask_gb * mask_gal * mask_sync

    if bwmask:
        mask = binmask(mask)

    ## trivial mask
    if nomask:
        mask[:] = 1

    print (f'coverage of mask_gb = ',np.average(mask_gb))
    print (f'coverage of mask_gal = ',np.average(mask_gal))
    print (f'coverage of mask_sync = ',np.average(mask_sync))
    print (f'coverage of mymask at nside:{nside0} = ',np.average(mask))

    mask = binmask(hp.ud_grade(mask, nside_out=nside))
    print (f'coverage of mymask at nside:{nside} = ',np.average(mask))

    if saveflg:
        hp.write_map(maskfname, mask)

    return np.array(mask)


def rms(data):
    return np.sqrt(np.mean(np.square(data)))


def RJ2CMB(T_RJ, nu):
    x = nu/56.78
    T_CMB = (np.exp(x) - 1)**2/(x**2 * np.exp(x)) * T_RJ

    return T_CMB


def CMB2RJ(T_CMB, nu):
    x = nu/56.78
    T_RJ = (x**2 * np.exp(x))/(np.exp(x) - 1)**2 * T_CMB

    return T_RJ


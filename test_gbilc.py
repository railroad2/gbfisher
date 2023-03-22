import numpy as np
import healpy as hp
import pylab as plt
import pysm

from itertools import combinations

from scipy.optimize import minimize
from pysm.nominal import models
from utils import *
from gbilc import *

try:
    from gbpipe.spectrum import get_spectrum_xpol
except:
    get_spectrum_xpol = get_spectrum_camb


def test():
    include_cmb = True
    include_noise = True#True
    nside = 8
    wp30 = 69
    wp90 = 93 
    wp145 = 47 
    wp220 = 292

    #gbilc_pysm(freqs=[145, 220], nside=nside, wps=[wp145, wp220], include_cmb=include_cmb, include_noise=include_noise, syncmask=False)
    gbilc_pysm(freqs=[145, 220, 30], nside=nside, wps=[wp145, wp220, wp30], include_cmb=include_cmb, include_noise=include_noise, syncmask=False)
    #gbilc_pysm(freqs=[145, 220, 30, 90], nside=nside, wps=[wp145, wp220, wp30, wp90], include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_class():
    include_cmb = True
    include_noise = True#True
    nside = 1024

    freqs = [40, 90, 150, 220]

    ## wps : pixel noise level in uK arcmin
    wps   = [39, 10, 15, 43]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_8ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_7ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 69, 82, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_9ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [ 145,  220, 30, 40, 90,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [  93,  584, 69, 82, 93, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_9ch_with_planck(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 353, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 620, 69, 82, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_8ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    #wps   = np.array([ 47, 292, 69, 82, 3600, 3600, 7200, 7200]) 
    wps   = np.array([ 93, 584, 69, 82, 3600, 3600, 7200, 7200]) 
    #wps   = np.array([ 57, 169, 69, 82, 3600, 3600, 7200, 7200]) 
    wps[0] = wps[0] #* 130./110
    wps[1] = wps[1] #* 130./110

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_7ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 69, 82, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_9ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [ 145,  220, 30, 40, 90,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    #wps   = [  93,  584, 69, 82, 93, 3600, 3600, 7200, 7200]
    wps   = [  47,  292, 69, 82, 47, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_9ch_with_planck_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 353, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 292, 620, 69, 82, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_10ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 90, 353, 30, 40,   11,   13,   17,   19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 292, 47, 620, 69, 82, 3600, 3600, 7200, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_4ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_4ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 292, 69, 82]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_2ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220]

    ## wps : pixel noise level in uK arcmin
    #wps   = [ 93.41, 590.45]
    wps   = [ 93, 584]

    cs_in = [1.475, -0.475]
    #cs_in = [1.345, -0.345]
    #cs_in = None
    include_fg = True

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, include_fg=include_fg,
               syncmask=True,
               verbose=True, plot_spectra=True, cs_in=cs_in)


def test_2ch_plot_spectrum(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584]

    cs_in = [1.485, -0.485]#None
    include_fg = True#True
    include_noise = False


    np.random.seed(124)
    include_cmb=True
    m_wicmb, masks = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, include_fg=include_fg,
               syncmask=True, return_cleaned_map=True,
               verbose=True, plot_spectra=False, cs_in=cs_in)

    np.random.seed(124)
    include_cmb=False
    m_wocmb, masks = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, include_fg=include_fg,
               syncmask=True, return_cleaned_map=True,
               verbose=True, plot_spectra=False, cs_in=cs_in)

    cs_in = [1.34, -0.34]
    np.random.seed(124)
    include_cmb=True
    m_cwicmb, masks = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, include_fg=include_fg,
               syncmask=True, return_cleaned_map=True,
               verbose=True, plot_spectra=False, cs_in=cs_in)

    np.random.seed(124)
    include_cmb=False
    m_cwocmb, masks = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, include_fg=include_fg,
               syncmask=True, return_cleaned_map=True,
               verbose=True, plot_spectra=False, cs_in=cs_in)

    fgs = gen_fg(freqs, nside)
    cl_fg145 = np.abs(get_spectrum_xpol(fgs[0], lmax=23, mask=masks)[0])
    cl_fg220 = np.abs(get_spectrum_xpol(fgs[1], lmax=23, mask=masks)[0])
   
    cl0 = get_spectrum_camb(lmax=100, tau=0.05, r=0.05, As=2.092e-9, isDl=False, CMB_unit='muK')
    np.random.seed(124)
    cmb = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    cl_cmb = np.abs(get_spectrum_xpol(cmb, lmax=23, mask=masks)[0]) 

    cl_wicmb = np.abs(get_spectrum_xpol(m_wicmb, lmax=23, mask=masks)[0])
    cl_wocmb = np.abs(get_spectrum_xpol(m_wocmb, lmax=23, mask=masks)[0])
    cl_cwicmb = np.abs(get_spectrum_xpol(m_cwicmb, lmax=23, mask=masks)[0])
    cl_cwocmb = np.abs(get_spectrum_xpol(m_cwocmb, lmax=23, mask=masks)[0])

    cl_cmb /= np.average(masks)
    cl_fg145 /= np.average(masks)
    cl_fg220 /= np.average(masks)
    cl_wicmb /= np.average(masks)
    cl_wocmb /= np.average(masks)
    cl_cwicmb /= np.average(masks)
    cl_cwocmb /= np.average(masks)

    nl93 = get_spectrum_noise(lmax=100, wp=93, isDl=False, CMB_unit='muK')
    nl584 = get_spectrum_noise(lmax=100, wp=584, isDl=False, CMB_unit='muK')
    nl250 = get_spectrum_noise(lmax=100, wp=250, isDl=False, CMB_unit='muK')
    nl300 = get_spectrum_noise(lmax=100, wp=300, isDl=False, CMB_unit='muK')

    ell23 = np.arange(21)+2
    ell100 = np.arange(99)+2

    plt.loglog(ell100, cl0[1].T[2:], 'k-', label='CMB theory', lw=0.5)
    plt.loglog(ell23, cl_cmb[1].T[2:], 'k*', label='CMB map', lw=0.5)
    #plt.loglog(ell23, cl_fg145[1].T[2:], 'k+--', label='foreground 145', lw=0.5)
    #plt.loglog(ell23, cl_fg220[1].T[2:], 'k+-.', label='foreground 220', lw=0.5)
    #plt.loglog(ell100, nl93[1].T[2:], 'y--', label='Noise 145 GHz (93 uK arcmin)')
    #plt.loglog(ell100, nl584[1].T[2:], 'y-.', label='Noise 220 GHz (584 uK arcmin)')
    plt.loglog(ell100, nl300[1].T[2:], 'g:', label='noise for fixed ILC (300 uK arcmin)')
    plt.loglog(ell100, nl250[1].T[2:], 'r:', label='noise for fit ILC (250 uK arcmin)')
    #plt.loglog(ell23, cl_wicmb[1].T[2:], 'g*--', label='cleaned map with fixed ILC', lw=2, ms=8)
    #plt.loglog(ell23, cl_cwicmb[1].T[2:], 'r*--', label='cleaned map with fit ILC', lw=2, ms=8)
    plt.loglog(ell23, cl_wocmb[1].T[2:], 'g*--', label='cleaned map with fixed ILC without CMB', lw=2, ms=8)
    plt.loglog(ell23, cl_cwocmb[1].T[2:], 'r*--', label='cleaned map with fit ILC without CMB', lw=2, ms=8)
    plt.xlabel(r"Multipole moment, $\ell$")
    plt.ylabel(r"$C_{\ell}^{EE} (\mu \mathrm{K}^2)$")
    plt.xlim(2, 30)
    plt.legend()

    #plt.show()


def test_3ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_3ch_ensemble(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 92.6, 584, 69]

    gbilc_pysm_ensemble(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)
    

def test_3ch_4xdet_org(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30 ]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 292, 69 ]
    #wps   = [ 93, 584, 69 ]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_3ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 30, 40]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 69, 82]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_5ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40, 90]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82, 93]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_5ch_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40, 90]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 292, 69, 82, 47]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_5ch_with_planck(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40, 353]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82, 620]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_5ch_with_planck_4xdet(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40, 353]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 47, 292, 69, 82, 620]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_planck(nside=8, include_cmb = True, include_noise = True):
    freqs = [30, 44, 70, 100, 143, 217, 353]

    ## wps : pixel noise level in uK arcmin
    wps   = [390.97, 487.76, 399.67, 166.84, 101.89, 154.16, 623.95]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_8ch_combination(nside=8, include_cmb=True, include_noise=True, cmb=[], fgs=[], idxused=np.arange(0,8)):
    freqs = np.array([145, 220, 30, 40,   11,   13,   17,   19, 353])

    ## wps : pixel noise level in uK arcmin
    wps   = np.array([ 93, 584, 69, 82, 3600, 3600, 7200, 7200, 623])

    freqs = freqs[idxused]
    wps = wps[idxused]

    f, cs, fgres, tnoise = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
                               include_cmb=include_cmb, include_noise=include_noise, 
                               cmb=cmb, fgs=np.array(fgs)[idxused], syncmask=False)

    return f, cs, fgres, tnoise


def makecomb(arr):
    idxs = []
    for i in range(2, len(arr)+1):
        idxs += combinations(arr, i) 

    return idxs


def combination_test():
    idxs0 = [0,1]
    others = [2, 3, 4, 5, 6, 7, 8]
    freqs = np.array([145, 220, 30, 40,   11,   13,   17,   19, 353])

    idxs1 = makecomb(idxs0+others)
    
    print (idxs1)
    f = []
    cs = []
    tnoise = []
    fgres = []

    cl0 = get_spectrum_camb(lmax=100, tau=0.05, r=0.05, As=2.092e-9, isDl=False, CMB_unit='muK')
    cmb = hp.synfast(cl0, nside=nside, verbose=0, new=1)
    fgs = gen_fg(freqs, nside=8)
    print (np.array(fgs))

    for i in idxs1:
        idxused = list(i)#idxs0 + list(i)
        f1, cs1, fgres1, tnoise1 = test_8ch_combination(nside, include_cmb, include_noise, cmb=cmb, fgs=fgs, idxused=idxused)

        f.append(f1)
        cs.append(cs1)
        tnoise.append(tnoise1)
        fgres.append(fgres1)
        print ('\n'+'*'*30+'\n')

    with open('combtest.dat', 'w') as ofile:
        ofile.write(f'freq, tnoise, cs')
        for f1, cs1, fgres1, tnoise1 in zip(f, cs, fgres, tnoise):
            print(f1, tnoise1, fgres1, cs1)
            ofile.write(f'{str(f1):36s} ')
            ofile.write(f'{tnoise1:10.3f} ')
            ofile.write(f'{fgres1:10.3e} ')
            ofile.write(f' [')
            for c in cs1:
                ofile.write(f'{c:10.8f} ')

            ofile.write(f'] \n')

    return 


def test_6ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40,   11,   17]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82, 3600, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_6ch_1(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40,   11,  19]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82, 3600, 7200]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_4ch(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 30, 40]

    ## wps : pixel noise level in uK arcmin
    wps   = [ 93, 584, 69, 82]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def test_wi90(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, 90]

    ## wps : pixel noise level in uK arcmin
    #wps   = [ 132, 584, 132, 69]#, 81]
    #wps   = [ 110, 780, 110 , 69]#, 81]
    #wps   = [ 156, 780, 156]#, 81]
    wps   = [ 110, 780, 110]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=True)


def test_wo90(nside=8, include_cmb = True, include_noise = True):
    freqs = [145, 220, ]#30]#, 40]

    ## wps : pixel noise level in uK arcmin
    #wps   = [ 93, 584]#, 69, 81]
    wps   = [ 110, 780, ]#69]#, 81]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=True)


def test_quijote(nside=8, include_cmb = True, include_noise = True):
    freqs = [30, 40, 11, 13, 17, 19]

    ## wps : pixel noise level in uK arcmin
    wps   = [113, 64, 2546, 2546, 3600, 3600]

    gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
               include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


def GBQT90(nside=8, include_cmb=True, include_noise=True):
    freqs = np.array([145, 220, 30, 40,   11,   13,   17,   19, 90])
    wps   = np.array([156, 780, 69, 82, 3600, 3600, 7200, 7200, 156])

    freqs = np.array([145, 220, 30, 40,   11,   13,   17,   19, ])
    wps   = np.array([110, 780, 69, 82, 3600, 3600, 7200, 7200, ])

    f, cs, fgres, tnoise = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
                               include_cmb=include_cmb, include_noise=include_noise, 
                               syncmask=False)

    return f, cs, fgres, tnoise


def gbilc_generaltest(freqs, wps, nside=8, syncmask=False):
    f, cs, fgres, tnoise = gbilc_pysm(freqs=freqs, nside=nside, wps=wps, 
                               include_cmb=include_cmb, include_noise=include_noise, 
                               syncmask=False)

    return f, cs, fgres, tnoise


if __name__=='__main__':
    nside = 8
    include_cmb = True
    include_noise = True#False # 

    test_wi90(nside, include_cmb, include_noise)





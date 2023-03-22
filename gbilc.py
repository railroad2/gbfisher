import numpy as np
import healpy as hp
import pylab as plt
import pysm

from itertools import combinations

from scipy.optimize import minimize
from pysm.nominal import models
from utils import *


try:
    from gbpipe.spectrum import get_spectrum_xpol
except:
    get_spectrum_xpol = get_spectrum_camb


def changefreq(map_in, f_in, f_out, beta):
    return (f_out/f_in)**beta * map_in


def masking(maps, mask):
    if mask is None:
        mm = maps
    else:
        mm = []
        mm.append(maps[0][mask==1])
        mm.append(maps[1][mask==1])
        mm.append(maps[2][mask==1])

    mm = np.array(mm)
    return mm


def ilcmaps(*maps, mask=None):
    if mask is None:
        mm = np.array(maps).copy()
    else:
        mm = []
        for m in maps:
            mm.append(masking(m, mask))

    def fnc(c):
        cs = list(c)
        cs.append(1-np.sum(c))
        cs = np.array(cs)
        tmp = [ct * mt for ct, mt in zip(cs, mm)]
        cleanedmap = np.sum(tmp, axis=0)
        #lk = np.var(cleanedmap[1]) + np.var(cleanedmap[2])
        lk = rms(np.sqrt(cleanedmap[1]**2 + cleanedmap[2]**2))
        return lk

    c0 = [1.0/(len(maps)-1)] * (len(maps)-1)
    res = minimize(fnc, c0)
    #print (res)
    cs = list(res.x)
    cs.append(1-np.sum(cs))
    tmp = [ct * mt for ct, mt in zip(cs, maps)]
    cleanedmap = sum(tmp)

    return res, cleanedmap


def show_rms(maps, mapname, mask=None):
    if len(np.shape(maps)) == 2:
        maps = [maps]
        mapname = [mapname]

    for m in maps:
        print ('*'*30, end=' ')

    print ('')
    for m, mn in zip(maps, mapname):
        print (f'* {mn:26} *', end=' ')

    print ('')
    for m in maps:
        mm = masking(m, mask)
        print (f' rms Q  = {rms(mm[1]):12.5E}'+' '*8, end=' ')

    print ('')
    for m in maps:
        mm = masking(m, mask)
        print (f' rms U  = {rms(mm[2]):12.5E}'+' '*8, end=' ')

    print ('')
    for m in maps:
        mm = masking(m, mask)
        print (f' rms Ip = {rms(np.abs(mm[1]+1j*mm[2])):12.5E}'+' '*8, end=' ')

    print ('')
    for m in maps:
        print ('*'*30, end=' ')

    print ('')

    return 


def gen_fg(freqs, nside=8):
    try:
        freqs = list(freqs)
    except: 
        freqs = list([freqs])

    dustmodel = 'd1'
    syncmodel = 's1'
    sky_config = {
        'synchrotron' : models(syncmodel, nside),
        'dust'        : models(dustmodel, nside),
        'freefree'    : models('f1', nside),
        #'cmb'         : models('c1', nside),
        'ame'         : models('a1', nside),
    }

    sky = pysm.Sky(sky_config)
    sky.output_unit = "uK_RJ"

    rot = hp.rotator.Rotator(coord=['g','c'])

    outdir = './maps/foregrounds/'

    fgs = []
    for freq in freqs:
        fwhm = 1 
        fweight = True if (nside > 16) else False 

        fg_gal = sky.signal()(freq)
        fg_equ = rot.rotate_map_alms(fg_gal, use_pixel_weights=fweight)
        fg_equ = RJ2CMB(fg_equ, freq)
        fg_equ_sm = hp.smoothing(fg_equ, fwhm=np.radians(fwhm), verbose=0)
        fgs.append(fg_equ_sm)

    if len(fgs) == 1:
        return fgs[0]

    return fgs

    instrument_delta_bpass = {
        'frequencies' : freqs,
        'channels' : (),
        'beams' : np.ones(3)*70.,
        'sens_I' : np.ones(3),
        'sens_P' : np.ones(3),
        'nside' : nside,
        'noise_seed' : 1234,
        'use_bandpass' : False,
        'add_noise' : False,
        'output_units' : 'uK_RJ',
        'use_smoothing' : True,
        'output_directory' : '.utofs/hive/home/kmlee/cmb/forecast/sim_map/',
        'output_prefix' : 'test',
    }

    instrument = pysm.Instrument(instrument_delta_bpass)
    instrument.observe(sky)

    return


def gbilc_pysm(freqs=[30, 90, 145, 220], nside=8, wps=[0, 0, 0, 0], 
               include_cmb=True, include_noise=True, include_fg=True,
               verbose=1, syncmask=False, cs_in=None, cmb=[], fgs=[],
               plot_maps=False, plot_spectra=False, return_cleaned_map=False):

    ## cmb
    print ('Generating CMB')
    if cmb == []:
        cl0 = get_spectrum_camb(lmax=100, tau=0.05, r=0.05, As=2.092e-9, isDl=False, CMB_unit='muK')
        cmb = hp.synfast(cl0, nside=nside, verbose=0, new=1)

    ## Foreground maps by pysm
    print ('Generating foregrounds')
    if fgs == []:
        fgs = gen_fg(freqs, nside)

    ## Noise maps
    print ('Generating Noise')
    nms = []
    for wp in wps: 
        nl0 = get_spectrum_noise(lmax=100, wp=wp, isDl=False, CMB_unit='muK')
        nm = hp.synfast(nl0, nside=nside, new=True, verbose=False)
        nms.append(nm)

    fgs = np.array(fgs)
    maps = fgs.copy()

    if not include_fg:
        maps *= 0.0

    for i, _ in enumerate(maps):
        if include_cmb:
            maps[i] += cmb
        if include_noise:
            maps[i] += nms[i]


    ## mask
    #print ('Generating Mask')
    #mask = mymask(nside=nside)#, syncmask=syncmask)
    mask = mymask(nside=nside, galmask=True, syncmask=syncmask)
    hp.mollview(mask)
    #plt.show()
    #mask *= 0
    #mask += 1

    ## ILC
    #print ('Fit')
    res, _ = ilcmaps(*maps, mask=mask)

    ## cleaned map
    if cs_in is not None:
        cs = cs_in
    else:
        cs = list(res.x)
        cs.append(1-np.sum(res.x))

    cs = np.array(cs)
    cleanedmap = sum([ct * mt for ct, mt in zip(cs, maps)])
    cleanednoise = sum([ct * nt for ct, nt in zip(cs, nms)])
    cleanedfg = sum([ct * fgt for ct, fgt in zip(cs, fgs)])

    tnoise = np.sqrt(np.sum(np.array(cs)**2 * np.array(wps)**2))
    #mask = mymask(nside=nside, galmask=True, syncmask=syncmask)
    print ('Result')
    if verbose:
        ## print rms
        show_rms([cmb], ['cmb (uK)'] , mask)
        for i, m in enumerate(maps):
            show_rms([maps[i], fgs[i], nms[i]], [f'map{freqs[i]} (uK)', f'foregrounds{freqs[i]} (uK)', f'noise{freqs[i]}(uK)'], mask)
        show_rms([cleanedmap, cleanedfg, cleanednoise], ['Cleaned map (uK)', 'Cleaned fg (uK)', 'Cleaned noise (uK)'], mask)

        print (f'coverage = {np.average(mask)}')

        print (f'coeffs (ILC) = {cs}')

        form1 = '{:8} |' + '{:10.2f}'* len(freqs)
        form  = '{:8} |' + '{:10.2f}'* len(freqs)
        print ('-' * 10*(1+len(freqs)))
        print (form1.format('freq', *freqs))
        print (form1.format('wp', *wps))
        print (form.format('coeff', *cs))
        print ('-' * 10*(1+len(freqs)))

        if plot_maps:
            maptobedrawn = cleanedmap*mask
            maptobedrawn = np.abs(maptobedrawn[1] + 1j*maptobedrawn[2])
            maptobedrawn[maptobedrawn==0] = hp.UNSEEN

            cmbip = np.abs(cmb[1] + 1j*cmb[2])
            cmbmin = min(cmbip)
            cmbmax = max(cmbip)

            if include_cmb:
                hp.mollview(maptobedrawn, title=f'Polarization intensity (cleaned map) {freqs} GHz', unit=r'$\mu K$')
                maptobedrawn = cleanedmap*mask
                maptobedrawn -= cmb*mask
                maptobedrawn = np.abs(maptobedrawn[1] + 1j*maptobedrawn[2])
                maptobedrawn[maptobedrawn==0] = hp.UNSEEN
                #hp.mollview(maptobedrawn, title=f'Polarization intensity (cleaned map - CMB) {freqs} GHz', min=cmbmin, max=cmbmax, unit=r'$\mu K$')
            else:
                hp.mollview(maptobedrawn, title=f'Polarization intensity (cleaned map) {freqs} GHz', min=cmbmin, max=cmbmax, unit=r'$\mu K$')
                pass

        masks = hp.smoothing(mask, np.radians(10))
        if plot_spectra:
            plt.figure()
            cl_cleaned = hp.anafast(cleanedmap*masks) / np.average(mask)
            cl_cleaned1 = get_spectrum_xpol(cleanedmap, lmax=23, mask=masks)[0]
            cl_cleaned1 = np.abs(cl_cleaned1)
            plt.loglog(cl0[:3].T, 'k', label='CMB')
            plt.loglog(cl_cleaned[:3].T, 'b-', label='cleaned_biased', lw=1)
            plt.loglog(cl_cleaned1[:3].T, 'r-', label='cleaned_unbiased')
            plt.legend()


        print (f'total noise level = {tnoise}')
        print ('-'*50)

    if return_cleaned_map:
        return cleanedmap, masks

    mm = masking(cleanedfg, mask)
    fgres = rms(np.abs(mm[1]+1j*mm[2]))

    #return cs
    return freqs, cs, fgres, tnoise


def gbilc_pysm_ensemble(freqs=[30, 90, 145, 220], nside=8, wps=[0, 0, 0, 0], 
                    include_cmb=True, include_noise=True, verbose=1, syncmask=False, cs_in=None):

    np.random.seed(0)
    ## cmb
    #print ('Generating CMB')
    cl0 = get_spectrum_camb(lmax=100, tau=0.05, r=0.05, As=2.092e-9, isDl=False, CMB_unit='muK')

    ## Foreground maps by pysm
    #print ('Generating foregrounds')
    fgs = gen_fg(freqs, nside)

    ## Noise maps
    #print ('Generating Noise')

    mask = mymask(nside=nside, galmask=False, syncmask=False)
    fgs = np.array(fgs)

    css = [] 
    ntest = 10
    for i in range(ntest):
        maps = fgs.copy()
        cmb = hp.synfast(cl0, nside=nside, verbose=0, new=True)#, fwhm=np.radians(0.6))

        nms = []
        for wp in wps: 
            nl0 = get_spectrum_noise(lmax=23, wp=wp, isDl=False, CMB_unit='muK')
            nm = hp.synfast(nl0, nside, new=True, verbose=False)#, fwhm=np.radians(0.6))
            nms.append(nm)

        for i, _ in enumerate(maps):
            if include_cmb:
                maps[i] += cmb
            if include_noise:
                maps[i] += nms[i]

        ## ILC
        res, _ = ilcmaps(*maps, mask=mask)

    ## cleaned map
        if cs_in is not None:
            cs = cs_in
        else:
            cs = list(res.x)
            cs.append(1-np.sum(res.x))

        cs = np.array(cs)
        print(cs)
        css.append(cs)

    print ("average")
    print (np.mean(css, axis=0))
    print (np.std(css, axis=0))
    return 
    
    #cleanedmap = sum([ct * mt for ct, mt in zip(cs, maps)])
    #cleanednoise = sum([ct * nt for ct, nt in zip(cs, nms)])
    #cleanedfg = sum([ct * fgt for ct, fgt in zip(cs, fgs)])

    print ('Result')
    if verbose:
        ## print rms
        show_rms([cmb], ['cmb (uK)'] , mask)
        for i, m in enumerate(maps):
            show_rms([maps[i], fgs[i], nms[i]], [f'map{freqs[i]} (uK)', f'foregrounds{freqs[i]} (uK)', f'noise{freqs[i]}(uK)'], mask)
        show_rms([cleanedmap, cleanedfg, cleanednoise], ['Cleaned map (uK)', 'Cleaned fg (uK)', 'Cleaned noise (uK)'], mask)

        print (f'coverage = {np.average(mask)}')

        print (f'coeffs (ILC) = {cs}')

        form1 = '{:8} |' + '{:10.2f}'* len(freqs)
        form  = '{:8} |' + '{:10.2f}'* len(freqs)
        print ('-' * 10*(1+len(freqs)))
        print (form1.format('freq', *freqs))
        print (form1.format('wp', *wps))
        print (form.format('coeff', *cs))
        print ('-' * 10*(1+len(freqs)))

        maptobedrawn = cleanedmap*mask
        maptobedrawn = np.abs(maptobedrawn[1] + 1j*maptobedrawn[2])
        maptobedrawn[maptobedrawn==0] = hp.UNSEEN

        cmbip = np.abs(cmb[1] + 1j*cmb[2])
        cmbmin = min(cmbip)
        cmbmax = max(cmbip)

        if include_cmb:
            #hp.mollview(maptobedrawn, title=f'Polarization intensity (cleaned map) {freqs} GHz', unit=r'$\mu K$')
            maptobedrawn = cleanedmap*mask
            maptobedrawn -= cmb*mask
            maptobedrawn = np.abs(maptobedrawn[1] + 1j*maptobedrawn[2])
            maptobedrawn[maptobedrawn==0] = hp.UNSEEN
            #hp.mollview(maptobedrawn, title=f'Polarization intensity (cleaned map - CMB) {freqs} GHz', min=cmbmin, max=cmbmax, unit=r'$\mu K$')
        else:
            #hp.mollview(maptobedrawn, title=f'Polarization intensity (cleaned map) {freqs} GHz', min=cmbmin, max=cmbmax, unit=r'$\mu K$')
            pass

        tnoise = np.sqrt(np.sum(np.array(cs)**2 * np.array(wps)**2))
        print (f'total noise level = {tnoise}')
        print ('-'*50)

    return cs


def test():
    include_cmb = True
    include_noise = True#True
    nside = 8
    wp30 = 69
    wp90 = 93 
    wp145 = 47 
    wp220 = 292

    gbilc_pysm(freqs=[145, 220, 30], nside=nside, wps=[wp145, wp220, wp30], include_cmb=include_cmb, include_noise=include_noise, syncmask=False)


if __name__=='__main__':
    test()
    plt.show()



import numpy as np
import healpy as hp
import pylab as plt

from fisher import *


def gb_quijote_9ch():
    tau0 = 0.05
    r0 = 0.05

    #wp = 105.5#79.26
    #wp = 67
    wp = 105 

    f_sky =0.222
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)

    plt.show()


def gb_2ch():
    tau0 = 0.05
    r0 = 0.05

    #wp = 105.5#79.26
    #wp = 67
    wp = 310#252.4#305.7

    f_sky =0.222
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)

    plt.show()


def gb_quijote_8ch():
    tau0 = 0.05
    r0 = 0.05

    #wp = 105.5#79.26
    #wp = 67
    wp = 303#116.6

    f_sky =0.226
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)

    plt.show()


def gb_quijote_planck_9ch():
    tau0 = 0.05
    r0 = 0.05

    wp = 132#302#153#22.17#115 

    f_sky =0.23
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)

    plt.show()


def planck():
    tau0 = 0.05
    r0 = 0.05

    wp = 176#89.43#99.52 

    f_sky = 0.226#1
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)

    plt.show()


def main():
    tau0 = 0.05
    r0 = 0.05

    #wp145 = 47.426
    #wp220 = 130.156
    wp145 = 47.426 #* np.sqrt(138) / np.sqrt(138*16)
    wp220 = 292 #* np.sqrt(23) / np.sqrt(23*16)
    wp0 = [wp145, wp220]

    c145 = 1.452
    wp1 = (1.452**2 * wp0[0]**2 + (1-1.452)**2 * wp0[1]**2) **0.5
    wp1 = 168#154 #79
    print (f"wp0 = {wp0}")
    print (f"wp1 = {wp1}")

    f_sky =0.222
    lmax = 23

    fisher_tau_r(tau0, r0, wp1, f_sky, lmax)

    plt.show()


def gb():
    tau0 = 0.05
    r0 = 0.00

    c145 = 1.468
    c220 = -0.468

    wp145pol = 108
    wp220pol = 780

    wp = (c145**2 * wp145pol**2 + c220**2 * wp220pol**2)**0.5 / 2**0.5
    #wp = 395 / 2**0.5

    print (f'   wp      = {wp} uK arcmin')
    print (f'   wp(pol) = {wp*2**0.5} uK arcmin')

    f_sky = 0.228
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)


def gb_90GHz():
    tau0 = 0.05
    r0 = 0.00

    c145 = -0.02
    c220 = -0.09
    c90  = 1.11

    wp145pol = 110
    wp90pol = 110
    wp220pol = 780

    freqs = [  145,   220,   90]
    cs    = [ 0.16, -0.14, 0.97] 
    wps   = [  156,   780,  156]
    #cs    = [-0.02, -0.09, 1.11] 
    #wps   = [  110,   780,  110]


    wp = np.sum(np.array(cs)**2 * np.array(wps)**2)**0.5 / 2**0.5
    #wp = (c145**2 * wp145pol**2 + c220**2 * wp220pol**2)**0.5 / 2**0.5

    print (f'   wp      = {wp} uK arcmin')
    print (f'   wp(pol) = {wp*2**0.5} uK arcmin')

    f_sky = 0.228
    lmax = 23

    fisher_tau_r(tau0, r0, wp, f_sky, lmax)


def gb_general(wps, cs, f_sky=0.23, lmax=23, tau0=0.05, r0=0.00):
    wp = np.sum(np.array(cs)**2 * np.array(wps)**2)**0.5 / 2**0.5

    print (f'   wp      = {wp} uK arcmin')
    print (f'   wp(pol) = {wp*2**0.5} uK arcmin')

    sigma = fisher_tau_r(tau0, r0, wp, f_sky, lmax)
    return sigma


if __name__=='__main__':
    gb_90GHz()

    plt.show()

import numpy as np
import healpy as hp
import pylab as plt

from utils import get_spectrum_camb


def confidence_ellipse(cov, ax, n_std=1.0, facecolor='none', **kwargs):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor='k')

    # Calculating the stdandard deviations from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)

    ellipse.set_transform(transf + ax.transData)

    ax.add_patch(ellipse)
    plt.autoscale()


def confidence_ellipse2(cov):
    phi = np.linspace(0, 2*np.pi, 100) 
    conflvl = 0.95
    r = 1.0#np.sqrt(5.991)
    eigval, eigvec = np.linalg.eig(cov)
    
    rmat = eigvec

    sigma0 = np.sqrt(cov[0,0])
    sigma1 = np.sqrt(cov[1,1])

    rx = r * np.cos(phi) * sigma0
    ry = r * np.sin(phi) * sigma1

    rx, ry = rmat.dot((rx, ry))

    plt.plot(rx, ry)


def Cmat_array(cl, nl):
    zero = np.zeros(len(cl[0]))
    Cmat = [ [cl[0]+nl[0], cl[3],       zero], 
             [cl[3],       cl[1]+nl[1], zero],
             [zero,        zero,        cl[2]+nl[2]] ]
    Cmat = np.array(Cmat)
    Cmat = np.transpose(Cmat, (2, 0, 1))

    return Cmat


def dCldtheta(Cl, tau0, nl):
    lmax = len(Cl)-1
    dtau = 0.00001
    Cln1 = Cmat_array(get_spectrum_camb(lmax, tau=tau0 - 1*dtau, isDl=False, CMB_unit='muK'), nl)
    Clp1 = Cmat_array(get_spectrum_camb(lmax, tau=tau0 + 1*dtau, isDl=False, CMB_unit='muK'), nl)
    dCldtau = (Clp1 - Cln1)/2/dtau
    return dCldtau


def d2Cldtheta2(Cl, tau0, r0, nl):
    lmax = len(Cl)-1
    dtau = 0.00001
    dr = 0.00001

    Clntau = Cmat_array(get_spectrum_camb(lmax, As=2.092e-9, tau=tau0 - 1*dtau, r=r0, isDl=False, CMB_unit='muK'), nl)
    Clptau = Cmat_array(get_spectrum_camb(lmax, As=2.092e-9, tau=tau0 + 1*dtau, r=r0, isDl=False, CMB_unit='muK'), nl)
    dCldtau = (Clptau - Clntau)/2/dtau

    Clnr = Cmat_array(get_spectrum_camb(lmax, As=2.092e-9, tau=tau0, r=r0 - 1*dr, isDl=False, CMB_unit='muK'), nl)
    Clpr = Cmat_array(get_spectrum_camb(lmax, As=2.092e-9, tau=tau0, r=r0 + 1*dr, isDl=False, CMB_unit='muK'), nl)
    dCldr = (Clpr - Clnr)/2/dr

    return dCldtau, dCldr


def fisher_tau(tau0):
    # F_ij = sum_l [ (2l+1)/2 * f_sky * Tr(C_l^-1 dC_l/dtheta_i C_l^-1 dC_l/dtheta_j) ]
    # theta = tau
    lmax = 11
    f_sky = 0.25
    wp = 10
    cl0 = get_spectrum_camb(lmax, tau=tau0, isDl=False, CMB_unit='muK')
    nl0 = noise_spectra(lmax, wp)
    ell = np.arange(lmax+1)
    Cls = Cmat_array(cl0, nl0)
    dCldtaus = dCldtheta(Cls, tau0, nl0)

    Fl = []
    for i, (Cl, dCldtau) in enumerate(zip(Cls, dCldtaus)):
        try:
            Clinv = np.linalg.inv(Cl)
        except np.linalg.LinAlgError:
            Clinv = Cl
        CCCC = Clinv.dot(dCldtau).dot(Clinv).dot(dCldtau) 
        Fl.append(np.trace(CCCC))

    F11 = np.sum((2*ell+1)/2 * f_sky * np.array(Fl))
    sigma1 = np.sqrt(1./F11)
    print (sigma1)


def noise_spectra(lmax, wp):
    # wp in uK arcmin 
    ell = np.arange(lmax+1)
    fwhm = np.radians(1)
    nl = (wp*np.pi/10800)**2 * np.exp(ell*(ell+1) * fwhm**2/8/np.log(2))

    nl = np.array([nl, nl*2, nl*2, nl*0])

    return nl


def fisher_tau_r(tau0, r0, wp, f_sky, lmax):
    # F_ij = sum_l [ (2l+1)/2 * f_sky * Tr(C_l^-1 dC_l/dtheta_i C_l^-1 dC_l/dtheta_j) ]
    # theta1 = tau
    # theta2 = r
    ell = np.arange(lmax+1)
    cl0 = get_spectrum_camb(lmax, As=2.092e-9, tau=tau0, r=r0, isDl=False, CMB_unit='muK')
    nl0 = noise_spectra(lmax, wp)

    Cls = Cmat_array(cl0, nl0)
    dCldtaus, dCldrs = d2Cldtheta2(Cls, tau0, r0, nl0)

    Fl11 = []
    Fl12 = []
    Fl21 = []
    Fl22 = []
    for i, (Cl, dCldtau, dCldr) in enumerate(zip(Cls, dCldtaus, dCldrs)):
        try:
            Clinv = np.linalg.inv(Cl)
        except np.linalg.LinAlgError:
            Clinv = Cl
        Fl11.append(np.trace(Clinv.dot(dCldtau).dot(Clinv).dot(dCldtau)))
        Fl12.append(np.trace(Clinv.dot(dCldtau).dot(Clinv).dot(dCldr)))
        Fl21.append(np.trace(Clinv.dot(dCldr).dot(Clinv).dot(dCldtau)))
        Fl22.append(np.trace(Clinv.dot(dCldr).dot(Clinv).dot(dCldr)))

    F11 = np.sum((2*ell+1)/2 * f_sky * np.array(Fl11))
    F12 = np.sum((2*ell+1)/2 * f_sky * np.array(Fl12)) 
    F21 = np.sum((2*ell+1)/2 * f_sky * np.array(Fl21))
    F22 = np.sum((2*ell+1)/2 * f_sky * np.array(Fl22))

    #F12 *= 0
    #F21 *= 0

    F = np.array([[F11, F12], [F21, F22]])
    cov = np.linalg.inv(F)
    sigma = np.sqrt(np.linalg.inv(F))
    cor = cov / np.outer((sigma[0,0], sigma[1,1]), (sigma[0,0], sigma[1,1]))

    print (f'Fisher Matrix:\n{F}')
    print (f'Covariance Matrix:\n{cov}')
    print (f'Correlation Matrix:\n{cor}')
    print (f'Error:\n{sigma}')
    print (f'Uncorrelated Error:\n{np.sqrt(1/F)}')
    print ('')
    print (f'tau = {tau0} +/- {sigma[0,0]}')
    print (f'  r = {r0} +/- {sigma[1,1]}')
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    confidence_ellipse(cov, ax, n_std=1)
    confidence_ellipse2(cov)

    return sigma


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


def gb_general(wps, cs, f_sky=0.23, lmax=23, tau0=0.05, r0=0.00):
    wp = np.sum(np.array(cs)**2 * np.array(wps)**2)**0.5 / 2**0.5

    print (f'   wp      = {wp} uK arcmin')
    print (f'   wp(pol) = {wp*2**0.5} uK arcmin')

    sigma = fisher_tau_r(tau0, r0, wp, f_sky, lmax)
    return sigma


if __name__=='__main__':
    main()


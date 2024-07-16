import time
import os
import skimage
import platform
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io as sio
import scipy.signal as signal
from tkinter.filedialog import askopenfilename, asksaveasfilename

def rdf_cal(RDF_parameter, mode='struct', data_trans=1, *args, **kwargs):
    # input data  #todo
    '''
    parameter list:
    element = [26, 15, 5, 29, 14]
    percentage = [100, 0.2, 0.2, 0.2, 0.2]

    phi_fit_begin = 1213  # todo
    phi_fit_end = 1557  # todo

    autodark_en = True

    smooth_range = 0.25
    pixel_begin = 60
    pixel_end = 1800
    pixel_adjust_b = -3
    pixel_adjust_e = 3
    pixel_adjust = 3 #  more
    smooth_en = True
    smooth_strength = 3
    polyfitN = 6

    L = 10
    rn = 2000
    # get data todo///
    dif = np.loadtxt('FePBCuSi_Pristine_XM 09 SAD.txt')

    #califactor =... todo
    DampingStrength = 0
    DampingStarPoint = 10000
    # calibration todo///
    califactor = 0.0021745
    '''
    if mode is 'dict':
        '''
        element = kwargs['element']
        percentage = kwargs['composition']
        phi_fit_begin = kwargs['phi_fit_begin']
        phi_fit_end = kwargs['phi_fit_end']
        autodark_en = kwargs['autodark_en']
        smooth_range = kwargs['smooth_range']
        pixel_begin = kwargs['pixel_begin']
        pixel_end = kwargs['pixel_end']
        pixel_adjust_b = kwargs['pixel_adjust_begin']
        pixel_adjust_e = kwargs['ixel_adjust_end']
        pixel_adjust = kwargs['pixel_adjust']
        smooth_en = kwargs['smooth_en']
        smooth_strength = kwargs['smooth_strength']
        polyfitN = kwargs['polyfitN']
        L = kwargs['L']
        rn = kwargs['rn']
        dif = kwargs['aver_inten']
        califactor = kwargs['califactor']
        Damping_en = kwargs['damping_en']
        DampingStrength = kwargs['damping_strength']
        DampingStarPoint = kwargs['damping_start_point']
        '''

        #dif = aver_inten
        #DampingStrength = damping_strength
        #DampingStarPoint = damping_start_point
        pass
    else:
        element = RDF_parameter.element
        percentage = RDF_parameter.composition
        phi_fit_begin = RDF_parameter.phi_fit_begin
        phi_fit_end = RDF_parameter.phi_fit_end
        autodark_en = RDF_parameter.autodark_en
        smooth_range = RDF_parameter.smooth_range
        pixel_begin = RDF_parameter.pixel_begin
        pixel_end = RDF_parameter.pixel_end
        pixel_adjust_b = RDF_parameter.pixel_adjust_begin
        pixel_adjust_e = RDF_parameter.pixel_adjust_end
        pixel_adjust = RDF_parameter.pixel_adjust
        smooth_en = RDF_parameter.smooth_en
        smooth_strength = RDF_parameter.smooth_strength
        polyfitN = RDF_parameter.polyfitN
        L = RDF_parameter.L
        rn = RDF_parameter.rn
        dif = RDF_parameter.aver_inten
        califactor = RDF_parameter.califactor
        Damping_en = RDF_parameter.damping_en
        DampingStrength = RDF_parameter.damping_strength
        DampingStarPoint = RDF_parameter.damping_start_point
    parameter_s = np.arange(0, np.size(dif))*califactor

    phi, background_phi = phi_calculation(dif, element, percentage, parameter_s, autodark_en=autodark_en,
                          phi_fit_begin=phi_fit_begin, phi_fit_end=phi_fit_end)

    RIF, yfit, phi_without_smooth, yfit_nosmo = modification_phi(phi, smooth_range,
                                                                 pixel_begin,
                                                                 pixel_end,
                                                                 parameter_s,
                                                                 pixel_adjust_b,
                                                                 pixel_adjust_e,
                                                                 pixel_adjust=pixel_adjust,
                                                                 smooth_en=smooth_en,
                                                                 smooth_strength=smooth_strength,
                                                                 polyfitN=polyfitN)
    if not Damping_en:
        DampingStrength = None
        DampingStarPoint = None
    G, G_corrected, G_corrected_nod, RIF_pristine, RIF_pristine_nos, r, RIF_damped = ft_rif(RIF, phi, parameter_s, yfit,
                                                               L, rn, califactor, pixel_begin,
                                                               pixel_end, pixel_adjust_b,
                                                               pixel_adjust_e, pixel_adjust,
                                                               DampingStrength, DampingStarPoint,
                                                               phi_without_smooth, yfit_nosmo)
    if data_trans:
        dict = {
            'background_phi': background_phi,
            'yfit': yfit,
            'yfit_nosmo': yfit_nosmo,
            'phi': phi,
            'G_corrected': G_corrected,
            'RIF_pristine': RIF_pristine,
            'r': r,
            'parameter_s': parameter_s,
            'pixel_end': pixel_end,
            'RIF_nosmo': RIF_pristine_nos,
            'G_corrected_nod': G_corrected_nod,
            'RIF_damped': RIF_damped
        }
        return dict
    else:
        window = plot_result(yfit, phi, G_corrected, RIF_pristine, r, parameter_s, pixel_end)
        return window


def camera_calibration(calibration=0.0021745, diff_length=1000, bins=1.0):
    """
    camera calibration value : how much reciprocal angstrom is in one pixel.

    Parameters
    ----------
    #type_of_camera
    calibration : float
       user-defined calibration factor.
       default using titan_245mm = 0.0021745
    diff_length : number
        length of sampling diffraction phi_function
    bins : number, optional
        combination the data with factor bins
    Returns
    -------
    parameter_s : ndarray
        reciprocal space base (X-axis)

    -------
    default calibration parameters for some cameras:
        califactor_titan_245mm = 0.0021745; % 245mm means Camera Length
        califactor_titan_195mm = 0.0018037;
        califactor_titan_480mm = 0.00111625;
        califactor_titan_upSTEM_CL245mm = 0.0087935;
        califactor_tecnai_ACOM_CL100 =  0.0068;
        califactor_merlinKarlsruhe_130 = 0.015757;
        califactor_tecnai_ACOM_CL100_new = 0.00655;
        califactor_tecnai_ACOM_CL80_new = 0.00841;
        califactor_ARM200_300mm = 0.001445;
        califactor_titan_480mm = 0.0010693;
        califactor_titan_300mm = 0.001776;
        califactor_muenster = 0.0039216;
        califactor_tecnai_SAD_CL150 = 0.001770;
        califactor_GRANDARM_JEOLDEMO = 0.006492;
        califactor_muenster = 0.0053014;
        califactor_NanJing_245mm = 0.002124;

    """

    calibration = calibration * bins
    parameter_s = calibration * np.arange(diff_length)
    return parameter_s


def phi_calculation(dif, element, percentage, parameter_s,
                    autodark_en=True, phi_fit_begin=0, phi_fit_end=2048):
    """
    prepare theoretical scattering background

    Parameters
    ----------
    dif : ndarray
        diffraction function
    element : ndarray
        atom number array of the element in the sample
    percentage : ndarray
        atom mass ratio in the sample
    parameter_s : ndarray
        reciprocal space base (X-axis)
    autodark_en : bool, optional
        Set to True to reduce the influence of dark noise
    phi_fit_begin : number, optional
        The start index of fitting range, default is 0
    phi_fit_end : number, optional
        The end index of fitting range, default is 2048
    Returns
    -------
    phi : ndarray
        phi function
    """
    f_mean = np.zeros(np.shape(parameter_s))
    f_sqrmean = np.zeros(np.shape(parameter_s))

    # prepare theoretical scattering background todo///
    try:
        enumerate(element)
    except:
        element = [element]
        percentage = [percentage]
    for i, item in enumerate(element):
        fit_para = ref_atom_para(item)
        parameter_s_sqr = np.multiply(parameter_s, parameter_s)
        f = (fit_para[0, 0] * np.power((parameter_s_sqr + fit_para[0, 1]), -1) +
            fit_para[0, 2] * np.power((parameter_s_sqr + fit_para[0, 3]), -1) +
            fit_para[1, 0] * np.power((parameter_s_sqr + fit_para[1, 1]), -1) +
            fit_para[1, 2] * np.exp(-1 * fit_para[1, 3] * parameter_s_sqr) +
            fit_para[2, 0] * np.exp(-1 * fit_para[2, 1] * parameter_s_sqr) +
            fit_para[2, 2] * np.exp(-1 * fit_para[2, 3] * parameter_s_sqr))
        f_mean = f_mean + percentage[i] * f
        f_sqrmean = f_sqrmean + percentage[i] * np.multiply(f, f)
    f_meansqr = np.multiply(f_mean, f_mean)

    # dark noise auto configuration

    if autodark_en:
        xopt = fit_background(dif, f_sqrmean, phi_fit_begin=phi_fit_begin, phi_fit_end=phi_fit_end)
        N = xopt[0]
        autoC = xopt[1]
    else:
        N = np.size(element)
        autoC = 0

    phi = np.multiply((dif-N*f_sqrmean-autoC)/(N*f_meansqr), parameter_s)
    background_phi = N*f_sqrmean+autoC
    return phi, background_phi


def ref_atom_para(atom_nr=1):
    """
    load atom parameters from the default files.

    Parameters
    ----------
    atom_nr: ndarray
        atomic numbers

    Return
    ------
    atom parameters, ndmatrix
        The essential parameters for calculating the theoretical scattering background

    """
    global rdf_atom_file
    try:
        rdf_atom_file
    except:
        try:  # platform.system() == 'Darwin'
            rdf_atom_file = os.getcwd()
            rdf_atom_file.replace('\\', '/')
            rdf_atom_file = rdf_atom_file + '/atom.mat'
            print(rdf_atom_file)
            data = sio.loadmat(rdf_atom_file)['fit_p']
        except:  # platform.system() == 'windows'
            rdf_atom_file = askopenfilename(initialdir=os.path.abspath('..'),
                               filetypes=(('MATLAB Data', '*.mat'), ("All Files", "*.*")),
                               title="Choose a atom mat file."
                               )
    data = sio.loadmat(rdf_atom_file)['fit_p']

    #path = os.path.abspath('..')
    #data = sio.loadmat(path+'/atom.mat')['fit_p']
    if atom_nr >= 1:
        return np.array(data[:, :, atom_nr-1])
    else:
        return np.zeros((3, 4))


def fit_background(dif, f_sqrmean, phi_fit_begin=0, phi_fit_end=2048):
    """
    using the downhill simplex algorithm to caculate the background parameter

    Parameters
    ----------
    dif : vector
        The to be fitted vector
    f_sqrmean : number
        mean of squared theoretical scattering background
    phi_fit_begin : number, optional
        The start index of fitting range, default is 0
    phi_fit_end : number, optional
        The end index of fitting range, default is 2048

    Returns
    -------
    xopt : ndarray
        Parameter that mininzes the LQ

    """
    xopt = scipy.optimize.fmin(cost_function_fit_background, [0, 0],
                               args=(dif, phi_fit_begin, phi_fit_end, f_sqrmean),
                               disp=False)
    return xopt


def cost_function_fit_background(parameters, profile, s_begin, s_end, f_sqrmean):

    background = parameters[0] * f_sqrmean + parameters[1]
    error2 = np.sum(np.power((profile[s_begin: s_end + 1] - background[s_begin: s_end + 1]), 2))
    return error2


def modification_phi(phi, smooth_range, pixel_begin, pixel_end, parameter_s,
                     pixel_adjust_b=0, pixel_adjust_e=0,
                     pixel_adjust=0, smooth_en=True,
                     smooth_strength=1, polyfitN=3, batch=False):
    """
    using cubic spline to smooth the phi in a large angle

    Parameters
    ----------
    phi : ndarray
        phi function, that to be smoothed
    smooth_range : float
        range of the phi function that to be smoothed
    pixel_begin : number
        first pixel of the phi, that to be smoothed
    pixel_end : number
        end pixel of the phi, that to be smoothed
    parameter_s : ndarray
        reciprocal space base (X-axis)
    pixel_adjust_b : number, optional
        adjust the pixel_begin
    pixel_adjust_e : number, optional
        adjust the pixel_end
    pixel_adjust : number, optional
        adjust the pixel_end
    smooth_en : bool, optional
        set True to enable smooth algorithm for the large angle
    smooth_strength : float, optional
        smoothing coefficient, default is 1.0
    polyfitN : int, optional
        polynomial fitting order, default is 3
    Returns
    -------
    RIF : ndarray
        smoothed phi
    yfit ; ndarray
        polynomial fitting for phi function
    """
    pixel_start = int(pixel_end * (1 - smooth_range))
    pixel_stop = pixel_end + pixel_adjust
    phi_without_smooth = None
    yfit_nosmo = None
    if smooth_en:
        phi_without_smooth = copy.deepcopy(phi)
        temp = signal.cspline1d(phi[pixel_start: pixel_stop + 1], smooth_strength)
        phi[pixel_start:pixel_stop + 1] = temp
        # different from matlab version

    start_point = [parameter_s[pixel_begin - 1 + pixel_adjust_b], phi[pixel_begin - 1 + pixel_adjust_b]]
    end_point = [parameter_s[pixel_end - 1 + pixel_adjust_e], phi[pixel_end - 1 + pixel_adjust_e]]
    if polyfitN > 0:
        p_coeffs = polyfix(x=parameter_s[pixel_begin - 1 + pixel_adjust_b: pixel_end + pixel_adjust_e],
                       y=phi[pixel_begin - 1 + pixel_adjust_b: pixel_end + pixel_adjust_e],
                       n=polyfitN,
                       xfix=[start_point[0], end_point[0]],
                       yfix=[start_point[1], end_point[1]])
        yfit = np.polyval(p_coeffs, parameter_s)
    else:
        yfit = phi * 0

    RIF = phi - yfit
    if smooth_en and not batch:
        start_point = [parameter_s[pixel_begin - 1 + pixel_adjust_b],
                       phi_without_smooth[pixel_begin - 1 + pixel_adjust_b]]
        end_point = [parameter_s[pixel_end - 1 + pixel_adjust_e],
                     phi_without_smooth[pixel_end - 1 + pixel_adjust_e]]
        if polyfitN > 0:
            p_coeffs_nosmo = polyfix(x=parameter_s[pixel_begin - 1 + pixel_adjust_b: pixel_end + pixel_adjust_e],
                           y=phi_without_smooth[pixel_begin - 1 + pixel_adjust_b: pixel_end + pixel_adjust_e],
                           n=polyfitN,
                           xfix=[start_point[0], end_point[0]],
                           yfix=[start_point[1], end_point[1]])
            yfit_nosmo = np.polyval(p_coeffs_nosmo, parameter_s)
        else:
            yfit_nosmo = phi * 0
    else:
        RIF_nosmo = None
    return RIF, yfit, phi_without_smooth, yfit_nosmo


def polyfix(x, y, n, xfix=[], yfix=[], xder=[], dydx=[]):
    """
    polyfix fit polynomial p to data, but specify value at specific points
    modified from matlab code polyfix

    Parameters
    ----------
    x : array_like
        Query points
    y : array_like
        Fitted values at query points
    n : int
        Degree of polynomial fit
    xfix : array_like, optional
        fixed query points
    yfix : array_like, optional
        fixed fitted values at xfix
    xder : array_like, optional
        fixed query points with derivative
    dydx : array_like, optional
        derivative value at xder

    Returns
    -------
    p : array_like
            coefficients of the polynomial
     """
    nfit = np.size(x)
    if not nfit == np.size(y):
        print('x and y must have the same size')
    nfix = np.size(xfix)
    if not nfix == np.size(yfix):
        print('xfix and yifx must have the same size')
    nder = np.size(xder)
    if not nder == np.size(dydx):
        print('xder and dydx must have the same size')

    nspec = nfix + nder
    specval = np.hstack((yfix, dydx))

    A = np.zeros((nspec, n+1))
    for i in range(0, n+1):
        A[0:nfix, i] = np.power(xfix, n-i)
    if nder > 0:
        for i in range(0, n):
            A[nfix:nfix+nder, i] = (n-i)*np.power(xder, n-i-1)
    if nfix > 0:
        lastcol = n + 1
        nmin = nspec - 1
    else:
        lastcol = n
        nmin = nspec
    if n < nmin:
        print('Polynomial degree too low. Cannot match all constraints')

    # Find the unique polynomial of degree nmin that fits the constraints.
    firstcol = n-nmin+1  # A(:,firstcol_lastcol) detrmines p0
    pc0 = np.linalg.solve(A[:, firstcol-1:lastcol], specval)  # Satifies A*pc = specval
    #  Now extend to degree n and pad with zeros:
    pc = np.zeros(n+1)
    pc[firstcol-1:lastcol] = pc0  # Satisfies A*pcfull = yfix
    # Column i in matrix X is the (n-i+1)'th power of x values
    X = np.zeros((nfit, n+1))
    for i in range(0, n+1):
        X[:, i] = np.power(x, n-i)
    yfit = y - np.polyval(pc, x)
    # We now find the p0 that mimimises (X*p0-yfit)'*(X*p0-yfit)
    # given that A*p0 = 0
    r, B = null(A)  # For any (n+1-nspc by 1) vector z, A*B*z = 0
    temp = np.dot(X, B.T)
    z = np.dot(np.linalg.pinv(temp), np.array([yfit]).T)  # Least squares solution of X*B*z = yfit
    p0 = np.dot(B.T, z)[:, 0]  # Satisfies A*p0 = 0;
    p = p0+pc  # Satisfies A*p = b;
    return p


def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:]


def ft_rif(RIF, phi, parameter_s, yfit, L, rn, califactor, pixel_begin, pixel_end,
           pixel_adjust_b, pixel_adjust_e, pixel_adjust, DampingStrength, DampingStartPoint,
           phi_without_smooth, yfit_nosmo):
    """
    FT of RIF to be PDF(reduced density function or radial

    Parameters
    ----------
    RIF : ndarray
        smoothed phi
    phi : ndarray
        phi function
    parameter_s : ndarray
        reciprocal space base (X-axis)
    fit : ndarray
        polynomial fitting for phi function
    L : int
        length of r in RDF
    rn : int
        pixel number in r (radius of RDF)
    califactor : float
        user-defined calibration factor
    pixel_begin : number
        end pixel of the phi, that to be smoothed
    pixel_end : number
        end pixel of the phi, that to be smoothed
    pixel_adjust_b : number, optional
        adjust the pixel_begin
    pixel_adjust_e : number, optional
        adjust the pixel_end
    pixel_adjust : number, optional
        adjust the pixel_end
    DampingStrength : float
        strength of damping
    DampingStarPoint : number
        from which pixel to start the damping

    Returns
    -------
    G : ndarray
        G(s) function (RDF without background subtraction)
    G_corrected : ndarray
        G(s) function (with background bustraction)
    RIF_pristine ; ndarray
        ???? todo
    """
    # r axis of distribution function G(r)

    r = np.linspace(0, L - L / rn, num=rn)  # define r in angstrom
    if DampingStartPoint is not None and DampingStartPoint < pixel_end:
        RIF_nod = copy.deepcopy(RIF)
        damp = np.exp(-1*DampingStrength*np.power(parameter_s, 2))
        damp[0:DampingStartPoint-1] = 1
        damp[DampingStartPoint - 1:pixel_end + pixel_adjust] = damp[DampingStartPoint-1:pixel_end+pixel_adjust] \
                                                               / damp[DampingStartPoint-1]
        RIF[DampingStartPoint-1:pixel_end+pixel_adjust] = np.multiply(RIF[DampingStartPoint-1:pixel_end+pixel_adjust],
                                                                     damp[DampingStartPoint-1:pixel_end+pixel_adjust])


        RIF_damped = RIF[:]
    else:
        RIF_damped = None

    phai = phi[pixel_begin-1+pixel_adjust_b: pixel_end+pixel_adjust_e]
    s = parameter_s[pixel_begin-1+pixel_adjust_b:pixel_end+pixel_adjust_e]
    sr_matrix = np.dot(np.array([s]).T, np.array([r]))
    G = 8*np.pi*np.dot(np.array([phai]), np.sin(2*np.pi*sr_matrix))*califactor

    G_corrected = 8*np.pi*np.dot(np.array(RIF[pixel_begin-1+pixel_adjust_b:pixel_end+pixel_adjust_e]),
                                 np.sin(2*np.pi*sr_matrix))*califactor

    if DampingStartPoint is not None and DampingStartPoint < pixel_end:
        G_corrected_nod = 8 * np.pi * np.dot(np.array(RIF_nod[pixel_begin - 1 + pixel_adjust_b:
                                                              pixel_end + pixel_adjust_e]),
                                             np.sin(2 * np.pi * sr_matrix)) * califactor
    else:
        G_corrected_nod = None
    RIF_pristine = phi[0:pixel_end+1+pixel_adjust_e] - yfit[0:pixel_end+1+pixel_adjust_e]
    if phi_without_smooth is not None:
        RIF_pristine_nos = phi_without_smooth[0:pixel_end+1+pixel_adjust_e] -\
                           yfit_nosmo[0:pixel_end+1+pixel_adjust_e]
    else:
        RIF_pristine_nos = None
    return G[0, :], G_corrected, G_corrected_nod, RIF_pristine, RIF_pristine_nos, r, RIF_damped


def plot_result(yfit, phi, G_corrected, RIF_pristine, r, parameter_s, pixel_end):
    window = plt.figure()
    plt.clf()
    plt.subplot(221)
    plt.subplots_adjust(left=0.10, bottom=0.10,
                        right=0.95, top=0.95,
                        wspace=0.25, hspace=0.25)

    plt.plot(parameter_s[0:pixel_end], phi[0:pixel_end])
    plt.plot(parameter_s[0:pixel_end], yfit[0:pixel_end])
    plt.title('G')
    plt.xlim((parameter_s[0], parameter_s[pixel_end-1]))
    plt.xlabel('Angstrom')
    plt.ylabel('Intensity')
    plt.grid(color='b', linestyle='-', linewidth=0.1)

    plt.subplot(222)
    plt.plot(RIF_pristine)
    plt.xlim((0, np.size(RIF_pristine)))
    plt.title('RIF_pristine')
    plt.xlabel('Angstrom')
    plt.ylabel('Intensity')
    plt.grid(color='b', linestyle='-', linewidth=0.1)

    plt.subplot(212)
    plt.plot(r, G_corrected)
    plt.xlim((r[0], r[-1]))
    plt.title('G_corrected')
    plt.xlabel('Angstrom')
    plt.ylabel('Intensity')
    plt.grid(color='b', linestyle='-', linewidth=0.1)
    #window.show()
    return window


if __name__ == '__main__':
    pass

import numpy as np

def tof2eKE(x, hv, s, t0, E0):
    """
    function for converting from TOF to energy
    requires:
    t the time of flight
    parameters:
        hv - photon energy
        s - source to detector distance
        t0 - start time
        E0 - potential offset in source
    """
    Me = 9.1093897e-31  # mass electron in kg
    ES = 6.242e18  # electrons per s

    eKE = (ES * 0.5 * Me) * ((1e9 * s / (x - t0))**2)  - E0

    return eKE

def tof2energy(x, hv, s, t0, E0):
    """
    function for converting from TOF to energy
    requires:
    t the time of flight
    parameters:
        hv - photon energy
        s - source to detector distance
        t0 - start time
        E0 - potential offset in source
    """
    Me = 9.1093897e-31  # mass electron in kg
    ES = 6.242e18  # electrons per s

    eKE = (ES * 0.5 * Me) * ((1e9 * s / (x - t0))**2)  - E0

    BE = hv - eKE
    return BE

def energy2tof(BE, hv, s, t0, E0):
    Me = 9.1093897e-31  # mass electron in kg
    ES = 6.242e18  # electrons per s   

    eKE = hv - BE
    tof = t0 + (1e9*s)/np.sqrt((2*(eKE + E0))/(ES * Me))

    return tof

    

def jacob(time, int, fit_params):
    '''
    get jacobian

    me*s^2/(t-t0)^3
    '''
    Me = 9.1093897e-31  # mass electron in kg

    jac = Me * (fit_params[1]**2)/((time-fit_params[2])**3)
    return (jac)*int

# def calib():
#     """
#     takes tof axis and counts axis and returns binding energy / jacobian corrected counts
#     This makes use of tof2energy and jacob functions
#     Also needs parameters from calibration fit
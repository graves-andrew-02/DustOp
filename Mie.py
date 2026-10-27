'''

This is a collection of functions to be used to calculate mie theory scattering
and absorption coefficients. 

'''

#IMPORTS AND STUFF
import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from numba import njit, jit
from tqdm import tqdm

#    General functions
###################################################################################################

def loglin_extrapolate(x, y, x_new):
  x3 = x_new
  x2 = x[-1]
  x1 = x[-2]

  y2 = y[-1]
  y1 = y[-2]
  y3 = y2*(y2/y1)**((x3-x2) / (x2-x1))

  return (y3)

  
  

#################################################
# interpolate a set of complex refractive indices
#################################################
def interp_m(wv, n, k, wv_range = [1e-5, 50], num_points = 200, wv_exact = [0.0]):
  if wv_range[0] <= 0.0 or wv_range[0] <= 0: # stop infinite x
    wv_range[0] = 1e-5

  if wv_range[1] > max(wv):
    print('WARNING: Max wavelength outside interpolation range - expect shananigans')
    print('Past limits of reference data, a slope is extrapolated to continue the set.')

    #find the slope at last few points

    n_new = loglin_extrapolate(wv, n, wv_range[1])
    k_new = loglin_extrapolate(wv, k, wv_range[1])

    wv = np.hstack((wv, np.array([wv_range[1]])))
    n = np.hstack((n,n_new))
    k = np.hstack((k, k_new))


  wv_range1 = np.log10(wv_range)
  interp_n = CubicSpline(wv, n)
  interp_k = CubicSpline(wv, k)

  wv_out = np.logspace(wv_range1[0], wv_range1[1], num_points)

  if len(wv_exact) != 1: # if you just want to use a set of wavelengths from a different file, for instance
    return wv_exact, interp_n(wv_exact) + 1j*interp_k(wv_exact)

  return wv_out, interp_n(wv_out) + 1j*interp_k(wv_out)



#    Jitted recursive functions
###################################################################################################

@njit
def compute_logarithmic_derivatives(z, n_max):
    """
    Computes D_n(z) = psi_n'(z) / psi_n(z) for n = 0 to n_max
    using downward recurrence
    """
    # Choose a higher starting point N_stop > n_max for convergence
    n_stop = int(max(n_max, abs(z)) + 50)

    # Initialize array
    D = np.zeros(n_stop + 1, dtype=np.complex128)

    # Seed top boundary condition
    D[n_stop] = 0.0 + 0.0j

    # Downward recurrence relation: D_{n-1}(z) = n/z - 1 / (D_n(z) + n/z)
    for n in range(n_stop, 0, -1):
        D[n - 1] = (n / z) - 1.0 / (D[n] + (n / z))

    return D[:n_max + 1]


@njit
def get_G_n_max(z, n_max):
  """
  use forward recursion to calculate G_n
  """
  # Initialize array
  G = np.zeros(n_max, dtype=np.complex128)

  G[0] = 0.0 + 1.0j
  # upwards recurrence relation: D_{n-1}(z) = n/z - 1 / (D_n(z) + n/z)
  for n in range(0, n_max-1):
      G[n+1] = - ((n+1) / z) + 1.0 / (((n+1) / z) - G[n])

  return G


#    Mie theory function for efficiencies and opacities
###################################################################################################

def get_coefficients(m, n_max, x, D):
    mx = m * x

    D_x = compute_logarithmic_derivatives(x, n_max)
    #initial zeroth-order ratio: k_0 = psi_0(x) / xi_0(x)
    k = 1j * np.exp(-1j * x) * np.sin(x)  # k_0 = psi_0 / xi_0
    g = 1/x - 1/np.tan(x)  # g_1 = psi_1 / psi_0
    l = 1/x - 1j           # l_1 = xi_1 / xi_0

    a = np.zeros(n_max, dtype=np.complex128)
    b = np.zeros(n_max, dtype=np.complex128)

    for n in range(1, n_max):
        # 1. Use current k_n (= psi_{n-1}/xi_{n-1}), g_n, and l_n
        K_a = (D[n]/m + n/x)
        a[n] = k * ((K_a * g - 1) / (K_a * l - 1))

        K_b = (D[n]*m + n/x)
        b[n] = k * ((K_b * g - 1) / (K_b * l - 1))

        # 2. Advance k to k_{n+1} = psi_n / xi_n for the next loop iteration
        k = k * (g / l)

        # 3. Advance g and l to order n+1
        # g_{n+1} = psi_{n+1} / psi_n using D_x at order n+1
        g = 1.0 / (D_x[n+1] + (n+1)/x)

        # l_{n+1} = xi_{n+1} / xi_n using upward recurrence
        l = (2*n + 1)/x - 1.0 / l

    return a, b




def efficiencies(m, wavelength, grain_radius, order=None): # efficiencies at single wavelength
    x = 2*np.pi * grain_radius / wavelength

    #in the base case of no order being specified,
    if order is None:
      x_max = np.max(x)
      order = int((x_max + 4*x_max**(1/3) + 2))

    #get all values for the logarithmic derivative
    D_mx = compute_logarithmic_derivatives(m*x, order)

    a, b = get_coefficients(m, order, x, D_mx)
    #initialise scattering and extinction coefficients
    Q_scatter    = 0.0
    Q_extinction = 0.0
    for n in range(1, order):
        #a_n, b_n = get_coefficients_n(m, n, x, D_mx[:,n])

        Q_scatter    += ((2*n + 1) * (abs(a[n])**2 + abs(b[n])**2))
        Q_extinction += ((2*n + 1) * (a[n] + b[n]).real)

    Q_scatter    *= wavelength**2 / (2*np.pi**2 *grain_radius**2)
    Q_extinction *= wavelength**2 / (2*np.pi**2 * grain_radius**2)
    return Q_scatter, Q_extinction




# get the scattering and absorption efficiencies for both scalar and array wavelengths
def get_efficiencies(m, wavelength, grain_radius, order=None):
  if isinstance(wavelength, np.ndarray):
    N = len(wavelength) 
    Q_scatter = np.zeros(N)
    Q_extinction = np.zeros(N)
    for i in range(0, N):
      Q_scatter[i], Q_extinction[i] = efficiencies(m[i], wavelength[i], grain_radius, order=None)
  else:
    Q_scatter, Q_extinction = efficiencies(m, wavelength, grain_radius, order=None)

  return Q_scatter, Q_extinction




def get_opacity(Q_scatter, Q_absorption, rho_grain, a):
  a_cm = a * 1e-4 # micrometer to cm as densities are usually in cgs
  if isinstance(a, np.ndarray):
    print('help')

  else:
    op_scatt =  Q_scatter    *0.75 / (a_cm *rho_grain)
    op_abs  = Q_absorption   *0.75 / (a_cm *rho_grain)
  
  return op_scatt, op_abs



#Random other functions for help with finding mean opacities
#################################################################################################


def dB_lambda_dT(wavelength_um, temp):
    '''
    Temperature derivative of blackbody spectral radiance.

    wavelength_um        ==        Wavelength in micrometres [um] - array
    temp                 ==             Temperature in kelvin (K) - float

    Returns
    array (same legnth as the wavelengths array)
        dB_lambda/dT in W sr^-1 m^-3 K^-1.
    '''

    if any(wavelength_um) <= 0 or temp <= 0:
        raise ValueError("Wavelength and temperature must be positive.")

    wavelength = wavelength_um * 1e-6  # µm -> m
    c = 2.99792458e8       # ms/s
    h = 6.62607015e-34     # J s
    k = 1.380649e-23       # J/K

    x = (h * c) / (wavelength * k * temp)

    # For very large x, the derivative is effectively zero.
    mask_large = x <=35000

    # Stable evaluation of exp(x) / (exp(x) - 1)^2

    exp_neg_x = np.exp(-x[mask_large])
    ratio = exp_neg_x / (1.0 - exp_neg_x)**2


    result = np.zeros(len(wavelength))
    result[mask_large] = ((2 / wavelength[mask_large]**6) * (1 / (temp**2))
        * ratio)

    return np.array(result)

def B_lambda(wavelength_um, temp):
    '''
    blackbody spectral radiance.

    wavelength_um == Wavelength in micrometres [um] - array
    temp          ==      Temperature in kelvin (K) - float

    Returns
    array (same legnth as the wavelengths array)

    THIS DOES NOT INCLUDE A CONSTANT OF 2hc^2 AT THE END, AS THIS CANCELS IN THE END
    '''

    if any(wavelength_um) <= 0 or temp <= 0:
        raise ValueError("Wavelength and temperature must be positive.")

    wavelength = wavelength_um * 1e-6  # µm -> m
    c = 2.99792458e8       # ms/s
    h = 6.62607015e-34     # J s
    k = 1.380649e-23       # J/K

    x = (h * c) / (wavelength * k * temp)

    # For very large x, the derivative is effectively zero.
    mask_large = x <=35000

    # Stable evaluation of exp(x) / (exp(x) - 1)^2

    exp_neg_x = np.exp(-x[mask_large])
    ratio = exp_neg_x / (1.0 - exp_neg_x)


    result = np.zeros(len(wavelength))
    result[mask_large] = (ratio / wavelength[mask_large]**5)

    return np.array(result)




########################################################
#.  Commonly used size distribution

def std_dist(r, P0 = 0.005):
  '''
  Reproduce the "standard size distribution" of Pollack et al (1985)

  radii are to be given in micrometers

  r == radius of dust grain [um] - array
  '''

  n = np.zeros(len(r))

  maska = np.where(r < P0)
  maskb = np.where((P0 <= r) & (r < 1.0))
  maskc = np.where((1.0 <= r) &  (r < 5.0))
  maskd = np.where(5.0 <= r)


  n[maska] = 1.0
  n[maskb] = (P0 / r[maskb])**3.5
  n[maskc] = ((1 / P0)**2)  * ((P0 / r[maskc])**5.5)
  n[maskd] = 0
  return n

def MRN(r, cutoff_low = 0.005, cutoff_high = 2.5, total = 1):
  '''
  Reproduce the "standard size distribution" of Mathis Rumpl and Nordsieck (1977)

  radii are to be given in micrometers

  r == radius of dust grain [um] - array
  '''

  n = np.zeros(len(r))

  mask = np.where((cutoff_low <= r) & (r < cutoff_high))
  n[mask] = r[mask]**(-3.5)

  #Normilize so integral goes to "total"
  n[mask] /= total * 2.5 * (cutoff_low**(-2.5)  -  cutoff_high**(-2.5))
  return n


def pwr_law(r, slope ,cutoff_low = 0.005, cutoff_high = 2.5, total = 1):
  '''
  Generalised power law function. only valid for negative power laws - will force input to negatives

  radii are to be given in micrometers

  r      ==   radius of dust grain [um]     - array
  slope  ==   power law of dist (negative)  - float
  '''

  used_slope = abs(float(slope))

  n = np.zeros(len(r))

  mask = np.where((cutoff_low <= r) & (r < cutoff_high))
  n[mask] = r[mask]**(-used_slope)

  #Normilize so integral goes to "total"
  n[mask] /= total * (used_slope-1.0) * (cutoff_low**(1.0-used_slope)  -  cutoff_high**(1-used_slope))
  return n


########################################################
#Stuff for making an interpolatable set of mie data

def create_mie_interp(wave_range, a_range, wv_raw, n_raw, k_raw, n_wv_points, n_a_points):
  '''
  Make a table of mie efficiencies for a range of grain sizes and wavelengths
  wave_range       ==   Range of wavelenths to cover  -   list of 2 floats
  a_range          ==
  wv_raw           ==
  n_raw            ==
  k_raw            ==
  n_points         ==
  '''
  wv, m = interp_m(wv_raw, n_raw, k_raw, wave_range, n_wv_points)
  a = np.logspace(np.log10(a_range[0]), np.log10(a_range[1]), n_a_points)

  vals = []
  for i in tqdm(range(0, n_a_points)):
    vals.append(get_efficiencies(m, wv, a[i]))

  vals_arr = np.array(vals)

  # Create 2D grid points of (a, wv)
  # We pair each grain size 'a' with each wavelength 'wv'
  A, WV = np.meshgrid(a, wv, indexing='ij')
  points = np.column_stack((A.ravel(), WV.ravel()))

  # Flatten the efficiency values corresponding to the points
  qsca_vals = vals_arr[:, 0, :].ravel()
  qext_vals = vals_arr[:, 1, :].ravel()

  # Create a single combined target array for both Qsca and Qext
  z = np.column_stack((qsca_vals, qext_vals))

  # Create a single multi-dimensional LinearNDInterpolator
  interp = LinearNDInterpolator(points, z)
  return interp, vals_arr, a, wv
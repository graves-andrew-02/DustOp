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

def extrapolate_power_law(x, y, n_fit=5, n_extra=20, x_end=None, direction="high",):
    """
    Extrapolate a power law using a linear fit in log-log space.

    Parameters
    ----------
    x, y : array-like
        Existing data. Must be positive for log-log extrapolation.
    n_fit : int
        Number of points at the selected end used to estimate the slope.
    n_extra : int
        Number of extrapolated points to generate.
    x_end : float or None
        Final x value of the extrapolation. If None, extend by the
        same logarithmic spacing as the existing data.
    direction : {"high", "low"}
        "high": extrapolate toward larger x.
        "low": extrapolate toward smaller x.

    Returns
    -------
    x_extended, y_extended : ndarray
        Original and extrapolated data, sorted by x.
    slope, intercept : float
        Fit parameters for log10(y) = slope*log10(x) + intercept.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays of equal length.")
    if len(x) < 2:
        raise ValueError("At least two data points are required.")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("All x and y values must be positive for log-log fitting.")
    if direction not in ("high", "low"):
        raise ValueError("direction must be 'high' or 'low'.")
    if not 2 <= n_fit <= len(x):
        raise ValueError("n_fit must be between 2 and the number of data points.")
    if n_extra < 1:
        raise ValueError("n_extra must be at least 1.")

    # Sort data by x and check for duplicate x values.
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("x values must be unique.")

    # Select the points at the end being extrapolated.
    if direction == "high":
        x_fit, y_fit = x[-n_fit:], y[-n_fit:]
        x_boundary = x[-1]
        if x_end is None:
            log_step = np.median(np.diff(np.log10(x)))
            x_end = 10 ** (np.log10(x_boundary) + log_step * n_extra)
        if x_end <= x_boundary:
            raise ValueError("For direction='high', x_end must exceed max(x).")
    else:
        x_fit, y_fit = x[:n_fit], y[:n_fit]
        x_boundary = x[0]
        if x_end is None:
            log_step = np.median(np.diff(np.log10(x)))
            x_end = 10 ** (np.log10(x_boundary) - log_step * n_extra)
        if x_end >= x_boundary or x_end <= 0:
            raise ValueError("For direction='low', x_end must be positive and below min(x).")

    # Fit log10(y) = slope*log10(x) + intercept.
    slope, intercept = np.polyfit(np.log10(x_fit), np.log10(y_fit), 1)

    # Generate new points logarithmically spaced beyond the data boundary.
    x_extra = np.geomspace(x_boundary, x_end, n_extra + 1)[1:]
    y_extra = 10 ** (slope * np.log10(x_extra) + intercept)

    x_extended = np.concatenate([x, x_extra])
    y_extended = np.concatenate([y, y_extra])
    return x_extended, y_extended, slope, intercept


  

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

    _, n, _,_ = extrapolate_power_law(wv, n, n_fit=5, n_extra=20, x_end=wv_range[1], direction="high",)
    wv, k, _,_ = extrapolate_power_law(wv, k, n_fit=5, n_extra=20, x_end=wv_range[1], direction="high",)

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


##############################################################
#   Mean opacity calculations
def get_rosseland(a, wave, n, T, interpolation, fractions = [0.53, 0.47], species = ['base']):
  '''
  get rosseland mean opacity

  input:
  a         == grain radii [um]   - array
  wave      == wavelength  [um]   - array
  n         == grain distribution - function (eg, MRN)
  T         == temperature        - float
  fractions == species fractions  - list/array
  species   == which dust to use  - list of strings

  '''


  if len(fractions) != len(species) and species[0] != 'base':
    #cant have that mate
    print('Dust fractions list and species list of differing length. Perhaps you missed an entry?')
    return

  # chose dust densities and decide what must have evaporated
  if species[0] == 'base':
    dust_bulk = [2.16] # graphite density in cgs units (Draine 1993, 2003)
    fractions = [1.0]
    #if T > 2273:
    #  fractions[0] = 0.0
#
    #if T > 4000:
    #  fractions[1] = 0.0

  else:
    if T > 2273 and any(species) == 'sic':
      fractions[species == 'sic'] = 0.0

    if T > 4000 and any(species) == 'gra':
      fractions[species == 'gra'] = 0.0


  ###########################
  #     CONSTANTS
  ###########################
  c = 2.99792458e10   # um/s
  h = 6.62607015e-27  # erg s
  k = 1.380649e-16    # erg/K


  #denominator is just what we have to divide by at the end, but here we want to do the integration over wavelength
  denominator = dB_lambda_dT(wave, T) # this will be consolidated into a scalar after integration

  avg_mass = 0.0
  avg_cross_section = 0.0

  for i in range(0, len(fractions)):
    avg_mass += fractions[i]*dust_bulk[i]

    # Evaluate Q_ext for every grain size and wavelength
    A, WV = np.meshgrid(a, wave, indexing='ij')
    coords = np.column_stack((A.ravel(), WV.ravel()))

    vals = interpolation(coords)
    Q_ext = vals[:, 1].reshape(A.shape)  # shape: (len(a), len(wave))

    # Integrate over grain size for each wavelength
    avg_cross_section += np.trapezoid(Q_ext * n(a)[:, None] * a[:, None]**2, a, axis=0)
    #avg_cross_section += np.trapezoid(mie_interp(wv,a) *n(a)*a**2, a)

  avg_mass *= np.trapezoid(n(a)*a**3, a)
  #print(0.75 *avg_cross_section / avg_mass)

  denominator_new = np.trapezoid(denominator* (avg_mass/avg_cross_section), wave)/0.75
  numerator = np.trapezoid(dB_lambda_dT(wave, T), wave)


  return 1e4 *numerator / denominator_new # the factor of 1e4 comes from conversion to cgs from um gs


def get_planck(a, wave, n, T, interpolation, fractions = [0.53, 0.47], species = ['base']):
  '''
  get planck mean opacity

  input:
  a         == grain radii [um]   - array
  wave      == wavelength  [um]   - array
  n         == grain distribution - function (eg, MRN)
  T         == temperature        - float
  fractions == species fractions  - list/array
  species   == which dust to use  - list of strings

  '''


  if len(fractions) != len(species) and species[0] != 'base':
    #cant have that mate
    print('Dust fractions list and species list of differing length. Perhaps you missed an entry?')
    return

  # chose dust densities and decide what must have evaporated
  if species[0] == 'base':
    dust_bulk = [2.16] # graphite density in cgs units (Draine 1993, 2003)
    fractions = [1.0]


  else:
    if T > 2273 and any(species) == 'sic':
      fractions[species == 'sic'] = 0.0

    if T > 4000 and any(species) == 'gra':
      fractions[species == 'gra'] = 0.0


  ###########################
  #     CONSTANTS
  ###########################
  c = 2.99792458e10   # um/s
  h = 6.62607015e-27  # erg s
  k = 1.380649e-16    # erg/K


  #denominator is just what we have to divide by at the end, but here we want to do the integration over wavelength
  B = B_lambda(wave, T) # this will be consolidated into a scalar after integration

  avg_mass = 0.0
  avg_cross_section = 0.0

  for i in range(0, len(fractions)):
    avg_mass += fractions[i]*dust_bulk[i]

    # Evaluate Q_ext for every grain size and wavelength
    A, WV = np.meshgrid(a, wave, indexing='ij')
    coords = np.column_stack((A.ravel(), WV.ravel()))

    vals = interpolation(coords)
    Q_ext = vals[:, 1].reshape(A.shape)  # shape: (len(a), len(wave))

    # Integrate over grain size for each wavelength
    avg_cross_section += np.trapezoid(Q_ext * n(a)[:, None] * a[:, None]**2, a, axis=0)

  avg_mass *= np.trapezoid(n(a)*a**3, a)

  # Correct integration and division
  numerator = np.trapezoid(avg_cross_section * B, wave) * 0.75 / avg_mass
  denominator = np.trapezoid(B, wave, axis=0)

  # 1e4 factor is the correct scale to convert from (um^2 / (g/cm^3 * um^3)) to (cm^2/g)
  return 1e4 * numerator / denominator

def get_planck_flipped(a, wave, n, T, interpolation, fractions = [0.53, 0.47], species = ['base']):
  '''
  get planck mean opacity

  input:
  a         == grain radii [um]   - array
  wave      == wavelength  [um]   - array
  n         == grain distribution - function (eg, MRN)
  T         == temperature        - float
  fractions == species fractions  - list/array
  species   == which dust to use  - list of strings

  '''


  if len(fractions) != len(species) and species[0] != 'base':
    #cant have that mate
    print('Dust fractions list and species list of differing length. Perhaps you missed an entry?')
    return

  # chose dust densities and decide what must have evaporated
  if species[0] == 'base':
    dust_bulk = [2.16] # graphite density in cgs units (Draine 1993, 2003)
    fractions = [1.0]


  else:
    if T > 2273 and any(species) == 'sic':
      fractions[species == 'sic'] = 0.0

    if T > 4000 and any(species) == 'gra':
      fractions[species == 'gra'] = 0.0


  ###########################
  #     CONSTANTS
  ###########################
  c = 2.99792458e10   # um/s
  h = 6.62607015e-27  # erg s
  k = 1.380649e-16    # erg/K


  #denominator is just what we have to divide by at the end, but here we want to do the integration over wavelength
  B = B_lambda(wave, T) # this will be consolidated into a scalar after integration

  avg_mass = 0.0
  avg_cross_section = 0.0

  for i in range(0, len(fractions)):
    avg_mass += fractions[i]*dust_bulk[i]

    # Evaluate Q_ext for every grain size and wavelength
    A, WV = np.meshgrid(a, wave, indexing='ij')
    coords = np.column_stack((A.ravel(), WV.ravel()))

    vals = interpolation(coords)
    Q_ext = vals[:, 1].reshape(A.shape)  # shape: (len(a), len(wave))

    # Integrate over grain size for each wavelength
    avg_cross_section += np.trapezoid(Q_ext * B, wave, axis = 1)

  avg_mass *= np.trapezoid(n(a)*a**3, a)

  # Correct integration and division
  numerator = np.trapezoid(avg_cross_section * n(a) * a**2, a, axis=0)* 0.75 / avg_mass
  denominator = np.trapezoid(B, wave, axis=0)

  # 1e4 factor is the correct scale to convert from (um^2 / (g/cm^3 * um^3)) to (cm^2/g)
  return 1e4 * numerator / denominator



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


def M_norm_pwr_law(r, slope ,cutoff_low = 0.005, cutoff_high = 2.5, total = 1):
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
  n[mask] /= total * (used_slope-4.0) * (cutoff_low**(4.0-used_slope)  -  cutoff_high**(4-used_slope))
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
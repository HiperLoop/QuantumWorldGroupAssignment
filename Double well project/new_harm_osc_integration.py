#%pylab inline
#from IPython.display import Image
'''
numerical solution is in the list
psi_low and psi_high
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq

sym = input("'even' or 'odd' eigenfunction? ")

if sym == 'even':
    psi_init = np.array([1.0, 0.0]) # even solution
elif sym == 'odd':
    psi_init = np.array([0.0, 1.0]) # odd solution

# integration range and steps
#
xmax = float(input("What is the max value of x [8]? ") or 8)
steps = int(input("What is the # of integration steps [2001]? ") or 2001)
#
x_start, x_end = 0.0, xmax

# initial bracketing energy values these are narrowed by brentq
#
en_low = float(input("Low bracket energy "))
en_up = float(input("High bracket energy "))

def U(x):
    return 0.25 * x**2

def integrate_ode(E, x_eval):
    """Integrate inward from x = 10 for a trial energy E."""
#    if E >= 0:
#        raise ValueError("A bound-state trial energy must be negative.")

    # Asymptotic decaying-tail initial conditions at x=10.
    psi_start =   psi_init[0]
    dpsi_start =  psi_init[1]

    def ode_rhs(x, psi):
        return [psi[1], (U(x) - E) * psi[0]]

    sol = solve_ivp(ode_rhs, (x_start, x_end), [psi_start, dpsi_start],
        t_eval=x_eval, method="DOP853", rtol=1e-9, atol=1e-11,)
        # DOP853 uses explicit, adaptive Runge–Kutta method of order 8
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def shooting_residual(E):
    """Residual for the hard-wall condition psi(0) = 0."""
    sol = integrate_ode(E, [x_end])
    return sol.y[0, -1]

# Lower and upper trial energies; both must be negative.

f_lower = shooting_residual(en_low)
f_upper = shooting_residual(en_up)
if f_lower * f_upper >= 0:
    raise ValueError(
        "The residuals do not bracket a root. Choose different energy bounds."
    )

# bracketing: core processes of the script
#
# repeatedly calls shooting_residual(E) and adjusts E
# until the residual is zero (within numerical tolerance).
#
en_value = brentq(shooting_residual, en_low, en_up, xtol=1e-10, )



# Integrate once more at the found energy to obtain the wavefunction.
x_range = np.linspace(x_start, x_end, steps)
#
sol = integrate_ode(en_value, x_range)

# Reverse arrays so x increases from 0 to 10, then normalize.
x = sol.t
psi = sol.y[0]
#psi /= np.sqrt(simpson(psi**2, x=x))

print(f"Ground state energy: E = {en_value:.10f}")
print(f"psi(0) = {psi[0]:.3e}")

# start figure settings
fig, ax = plt.subplots()
#
ax.set_xlim([-xmax, xmax])  # use xmax in the flat region for final plot
if sym == 'even':
    ax.set_ylim([-1.4, 1.4])
elif sym == 'odd':
    ax.set_ylim([-0.9, 0.9])
    
ax.set_title('Q.H.O.');
#
ax.spines['right'].set_color('none') # no vertical right border
ax.spines['top'].set_color('none') # no top border
ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0
# end figure settings
        
ax.plot(x, psi,'C0')   # the plots are the same
if sym == 'even':
    ax.plot(- x_range, psi,'C0')
elif sym == 'odd':
    ax.plot(- x_range, - psi,'C0')


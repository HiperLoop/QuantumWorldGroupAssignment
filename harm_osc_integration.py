#%pylab inline
#from IPython.display import Image
'''
numerical solution is in the list
psi_low and psi_high
'''
import numpy as np
import math
from scipy.integrate import odeint, ode
from scipy.integrate import quad, dblquad, tplquad
import scipy.special as sp
import matplotlib
import matplotlib.pyplot as plt

# initial conditions for psi and psi' at $t=0$
# list psi: psi[0] = psi, psi[1]=psi'

# initial conditions for even solutions

analytic_or_numerical = input("'analytic' or 'numerical' for exact solution or numerical approximation")

# integration range and steps
#
xmax = float(input("What is the max value of x [8]? ") or 8)
#xmax = 8   # adjust according to the width of psi

if analytic_or_numerical == 'analytic':
    n = int(input("For what quantum number 'n' is the solution supposed to be calculated?"))

    x_range = np.linspace(0,xmax, 100)

    def psi_eval(x: float):
        normalization = 1.0 / (np.pi**0.25 * np.sqrt((2.0**n) * math.factorial(n)))
    
        H_n = sp.eval_hermite(n, x)
        
        psi = normalization * H_n * np.exp(-x**2 / 2.0)
    
        return psi

    # start figure settings
    fig, ax = plt.subplots()

    ax.set_xlim([-xmax, xmax])  # use xmax in the flat region for final plot
    ax.set_ylim([-1.2, 1.2])

    ax.set_title('Q.H.O.');

    ax.spines['right'].set_color('none') # no vertical right border
    ax.spines['top'].set_color('none') # no top border
    ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
    ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0

    ax.plot(x_range, psi_eval(x_range))
    ax.plot(-x_range, ((-1)**n) * psi_eval(x_range))
    

elif analytic_or_numerical == 'numerical':
    sym = input("'even' or 'odd' eigenfunction? ")

    if sym == 'even':
        psi = np.array([1.0,0.0]) # even solution
    elif sym == 'odd':
        psi = np.array([0.0,1.0]) # odd solution

    # number of iterations while narrowing the energy bracket
    # increase if a solution is not yet
    #
    iterations = int(input("What is the # of iterations [16]? ") or 16)
    #iterations = 20

    # number of integration steps in range [0,xmax]
    #
    steps = int(input("What is the # of integration steps [100]? ") or 100)
    #steps = 100

    # initial bracketing energy values
    # these are narrowed after each iteration
    en = [0.0,0.0]  # creates list for bracketing energies
    #
    en[0] = float(input("Low bracket energy "))
    en[1] = float(input("High bracket energy "))

    norm = input("Normalise wave function? Y/n")
    normalise = norm.lower() != 'n'

    x_range = np.linspace(0,xmax,steps)   # all integration steps

    def fct(psi, x):
        """
        The right-hand side of the ODE
        """
        return [psi[1], (x**2 - (energy * 2)) * psi[0]]

    # start figure settings
    fig, ax = plt.subplots()

    ax.set_xlim([-xmax, xmax])  # use xmax in the flat region for final plot
    ax.set_ylim([-1.2, 1.2])

    ax.set_title('Q.H.O.');

    ax.spines['right'].set_color('none') # no vertical right border
    ax.spines['top'].set_color('none') # no top border
    ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
    ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0
    # end figure settings

    last = [0.0,0.0] # used to store final values of psi

    for loop in range(0,iterations):
        # integration for lower bracket energy
        i = 0
        energy = en[i] # energy value for each iteration
        psi_low = odeint(fct,psi,x_range) # returns numerical solution
        # plot solution at end of iterations
        if loop == iterations - 1:
            ax.plot(x_range, psi_low[:,0])
            if sym == 'even':
                ax.plot(- x_range, psi_low[:,0])
            elif sym == 'odd':
                ax.plot(- x_range, - psi_low[:,0])
        last[i] = psi_low[-1][0]

        # integration for higher bracket energy    
        i = 1
        energy = en[i]
        psi_high = odeint(fct,psi,x_range)
        if loop == iterations - 1:     # plot again to also check visually that
            ax.plot(x_range, psi_high[:,0])   # the plots are the same
            if sym == 'even':
                ax.plot(- x_range, psi_low[:,0])
            elif sym == 'odd':
                ax.plot(- x_range, - psi_low[:,0])
        last[i] = psi_high[-1][0]

        # if final values of psi have different signs the solution 
        # is bracketed in the interval, so integrate again
        # using the middle energy
        if last[0]*last[1] < 0:
            energy = 0.5 * (en[0] + en[1])
            psi_mid = odeint(fct,psi,x_range)
            mid_last = psi_mid[-1][0]
            
            if last[0]*mid_last < 0:
                en[1] = energy
            elif last[1]*mid_last < 0:
                en[0] = energy      
        else:
            print('energies not bracketing a solution')
            break

        # print bracket energies & their different at the last iteration
        if loop == iterations - 1: 
            print("final bracket =", en)
            print("bracket width =", en[1] - en[0])

    if normalise:
        delta_x: float = x_range[1] - x_range[0]
        normalisation_constant: float = np.sqrt(2 * (delta_x * np.sum(((psi_low[:, 0] + psi_high[:, 0]) * 0.5)**2)))
        psi = ((psi_low[:, 0] + psi_high[:, 0]) * 0.5) / normalisation_constant

        ax.clear()

        ax.set_xlim([-xmax, xmax])  # use xmax in the flat region for final plot
        ax.set_ylim([-1.2, 1.2])

        ax.set_title('Q.H.O., n = 5, normalised');

        ax.spines['right'].set_color('none') # no vertical right border
        ax.spines['top'].set_color('none') # no top border
        ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
        ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0
        ax.plot(x_range, psi)
        ax.plot(-x_range, -psi)

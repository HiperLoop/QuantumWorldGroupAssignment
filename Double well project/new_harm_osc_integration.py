import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import brentq


def all(sym, xmax, steps, en_low, en_up, pos, col, prev_value, double, enn):
    # Setup plot
    ax = plt.subplot(N//2 + 1, 2, pos)

    # Initial solver values
    x_start, x_end = xmax, 0.0

    # Integration function
    def integrate_ode(E, x_eval):
        """Integrate inward from xmax to 0 for a trial energy E."""
        psi_start = 1e-12
        k_tail = np.sqrt(max(0.0, 2 * (-xmax**2 + 0.045*xmax**4 - E)))
        dpsi_start = -k_tail * psi_start

        # RHS of the Schrodinger equation
        def ode_rhs(x, psi):
            return [psi[1], 2 * (-x**2 + 0.045*x**4 - E) * psi[0]]

        # scipy numerical solution
        sol = solve_ivp(ode_rhs, (x_start, x_end), [psi_start, dpsi_start],
            t_eval=x_eval, method="DOP853", rtol=1e-12, atol=1e-14)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def shooting_residual(E):
        """Residual at x = 0: psi'(0)=0 for even, psi(0)=0 for odd."""
        sol = integrate_ode(E, [x_end])
        return sol.y[1, -1] if sym == 'even' else sol.y[0, -1]

    # Find more percise energy from initial bracket
    e_grid = np.linspace(en_low, en_up, 41)
    res_grid = [shooting_residual(e_val) for e_val in e_grid]
    for i in range(len(e_grid) - 1):
        if res_grid[i] * res_grid[i + 1] < 0:
            en_low, en_up = e_grid[i], e_grid[i + 1]
            break
    f_lower = shooting_residual(en_low)
    f_upper = shooting_residual(en_up)
    if f_lower * f_upper >= 0:
        raise ValueError(
            "The residuals do not bracket a root. Choose different energy bounds."
        )

    en_value = brentq(shooting_residual, en_low, en_up, xtol=1e-13)

    x_range = np.linspace(x_start, x_end, steps)
    sol = integrate_ode(en_value, x_range)

    x = sol.t[::-1]
    psi = sol.y[0, ::-1]

    print(f"Ground state energy: E = {en_value:.10f}")
    print(f"psi(0) = {psi[0]:.3e}")

    # start figure settings
    ax.set_xlim([-xmax, xmax])
    if double:
        ax.set_title(f'state with n = {2 * enn}, {2 * enn + 1}\neven energy: E = {prev_value:.10f}\nodd energy E: = {en_value:.10f}')
    elif sym == 'odd':
        ax.set_title(f'state with n = {enn}\nodd energy E: = {en_value:.10f}')
    else:
        ax.set_title(f'state with n = {enn}\neven energy: E = {en_value:.10f}')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    # end figure settings

    # Normalise wave function
    normalisation_constant = np.sqrt(2.0 * simpson(psi**2, x=x))
    psi = psi / normalisation_constant

    # Plot wave functions
    ax.plot(x, psi, col)
    if sym == 'even':
        ax.plot(-x, psi, col)
    elif sym == 'odd':
        ax.plot(-x, -psi, col)

    return en_value

# User input
xmax = float(input("What is the max value of x [6]? ") or 6)
steps = int(input("What is the # of integration steps [2001]? ") or 2001)
N_start = int(input("What energy state index to start at? [0]") or 0)
N_end = int(input("What energy state index to end at? [2]") or 2)
N = N_end - N_start + 1

# Energy estimates
def e(n):
    return int(-(50/9) + 2*(n + 0.5)) - 0.5
def true_e(n):
    return -(50/9) + 2*(n + 0.5)

# Plotting
fig = plt.figure()
fig.set_size_inches(12, 8)
fig.subplots_adjust(hspace=0.55, wspace=0.25)
plt.tight_layout()
e_evens = []
e_odds = []
e_bracket_width = 1

# Do numerical solving and print results
for n in range(N):
    n = N_start + n
    if(e(n) < 0):
        e_evens.append(all("even", xmax, steps, e(n) - e_bracket_width, e(n) + e_bracket_width, n+1-N_start, "#ec7104", 0, True, n-N_start))
        print(f"Even theoretical e diff = {e_evens[n-N_start] - true_e(n)}")
        e_odds.append(all("odd", xmax, steps, e(n) - e_bracket_width, e(n) + e_bracket_width, n+1-N_start, "#04b2ec", e_evens[n-N_start], True, n-N_start))
        print(f"Odd theoretical e diff = {e_odds[n-N_start] - true_e(n)}")
        print(f"Split energy = {np.abs(e_evens[n-N_start] - e_odds[n-N_start])}")
    else:
        e_bracket_width = 0.4
        all("even", xmax, steps, e(n) - 4*e_bracket_width, e(n) + e_bracket_width, 2*(n-N_start) + 1, "#ec7104", 0, False, 2*(n))
        all("odd", xmax, steps, e(n) - 2*e_bracket_width, e(n) + e_bracket_width, 2*(n-N_start) + 2, "#04b2ec", 0, False, 2*(n) + 1)

plt.show()
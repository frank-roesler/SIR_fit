import numpy as np

class SIR_fit():
    """Contains a collection of methods to simulate data of an infectious disease outbreak and fit a simple SIR model"""
    def __init__(self, dt=0.01, T=200, S0=1, I0=0.01, R0=0, N_times=100):
        self.dt      = dt       # Length of numerical time steps
        self.T       = T        # Length of time interval
        self.S0      = S0       # Initial value for "Susceptible"
        self.I0      = I0       # Initial value for "Infected"
        self.R0      = R0       # Initial value for "Removed"
        self.N_times = N_times  # Times at which measurements are taken

    def compute_infectious(self, beta, gamma):
        """Computes rate of infectious individuals from parameters beta, gamma"""
        numpoints = int(np.ceil(self.T / self.dt))
        S = np.zeros(numpoints)
        I = np.zeros(numpoints)
        R = np.zeros(numpoints)
        S[0] = self.S0
        I[0] = self.I0
        R[0] = self.R0
        for i in range(numpoints - 1):
            S[i + 1] = S[i] - beta * self.dt * S[i] * I[i]
            I[i + 1] = I[i] + beta * self.dt * S[i] * I[i] - gamma * self.dt * I[i]
            R[i + 1] = R[i] + gamma * self.dt * I[i]
        return I

    def compute_full_model(self, beta, gamma):
        """Computes S, I, R from parameters beta, gamma"""
        numpoints = int(np.ceil(self.T / self.dt))
        S = np.zeros(numpoints)
        I = np.zeros(numpoints)
        R = np.zeros(numpoints)
        S[0] = self.S0
        I[0] = self.I0
        R[0] = self.R0
        for i in range(numpoints - 1):
            S[i + 1] = S[i] - beta * self.dt * S[i] * I[i]
            I[i + 1] = I[i] + beta * self.dt * S[i] * I[i] - gamma * self.dt * I[i]
            R[i + 1] = R[i] + gamma * self.dt * I[i]
        return S,I,R

    def generate_data(self, beta, gamma):
        """Generates simulated data I+(gaussian noise)"""
        numpoints = int(np.ceil(self.T / self.dt))
        numeric_grid = np.linspace(0,self.T,numpoints)
        measurement_times = np.linspace(0,self.T,self.N_times)
        initial_curve = self.compute_infectious(beta, gamma)
        I_simulated = np.interp(measurement_times, numeric_grid, initial_curve)
        noise = np.random.randn(len(I_simulated)) / 30 * np.sqrt(I_simulated)
        I_simulated += noise
        I_simulated[I_simulated < 0] = 0

        return I_simulated

    def square_error(self, beta, gamma, data):
        """computed chi^2 error between data and model"""
        model = self.compute_infectious(beta, gamma)
        numpoints = int(np.ceil(self.T / self.dt))
        numeric_grid = np.linspace(0, self.T, numpoints)
        measurement_times = np.linspace(0, self.T, self.N_times)
        interpolated_model = np.interp(measurement_times, numeric_grid, model)
        e = np.nansum((data - interpolated_model) ** 2)
        return e

    def gradient(self, f, p, eps=0.001):
        """computes gradient of chi^2 error between data and model"""
        x = p[0]
        y = p[1]
        f_x = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
        f_y = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
        return np.array([f_x, f_y])

    def hessian(self, f, p, eps=0.001):
        """computes Hessian of chi^2 error between data and model"""
        x = p[0]
        y = p[1]
        f_xx = (f(x + eps, y) + f(x - eps, y) - 2 * f(x, y)) / (eps ** 2)
        f_yy = (f(x, y + eps) + f(x, y - eps) - 2 * f(x, y)) / (eps ** 2)
        f_xy = (f(x + eps, y + eps) + f(x - eps, y - eps) - f(x - eps, y + eps) - f(x + eps, y - eps)) / (4 * eps ** 2)
        return np.array([[f_xx, f_xy], [f_xy, f_yy]])


    def gradient_step(self, data, stepsize, beta, gamma):
        """Performs one step of gradient descent starting from
        the point [beta, gamma] with given stepsize"""
        def cost_function(beta, gamma):
            return self.square_error(beta, gamma, data)

        dx, dy = self.gradient(cost_function, [beta, gamma])
        abs_grad_chi = np.sqrt(dx ** 2 + dy ** 2)
        beta_new = beta - stepsize * dx / abs_grad_chi
        gamma_new = gamma - stepsize * dy / abs_grad_chi

        return beta_new, gamma_new

    #-------------------------------------------------------------------------------------------------------------------
    # Conjugate Gradient:
    #-------------------------------------------------------------------------------------------------------------------

    def conjugate_gradient(self, f, p0, eps=0.01, imax=10, jmax=10, CG_tol=1e-5, N_tol=1e-6, sigma0=0.05):
        """Minimizes a function f(x,y) via conjugate gradient method with initial guess p0"""
        i = 0
        k = 0
        x = np.array(p0)
        r = -self.gradient(f, x, eps)
        M = self.hessian(f, x, eps)
        s = np.linalg.solve(M,r)
        d = s
        delta_new = r @ d
        delta_0 = delta_new
        residue=[f(*p0)]
        while i < imax and delta_new > delta_0 * CG_tol ** 2:
            j = 0
            delta_d = d@d
            alpha = -sigma0
            grad_f = self.gradient(f, x+sigma0*d, eps)
            eta_prev = grad_f@d
            while j < jmax and alpha**2 * delta_d > N_tol ** 2:
                eta = self.gradient(f, x, eps)@d
                alpha = alpha*eta/(eta_prev-eta)
                x = x+alpha*d
                eta_prev = eta
                j+=1
                # residue.append(f(*x))
                # print('j=',j)
            r = -self.gradient(f, x, eps)
            delta_old = delta_new
            delta_mid = r@s
            M = self.hessian(f, x, eps)
            s = np.linalg.solve(M, r)
            delta_new = r @ s
            Beta = (delta_new-delta_mid) / delta_old
            k += 1
            if k == 2 or Beta <= 0:
                d = s
                k = 0
            else:
                d = s+Beta*d
            i += 1
            print(i)
        return x, residue

    def fit_model_CG(self, data):
        """Computes beta, gamma via conjugate gradient method"""
        # Initialize parameters randomly:
        beta, gamma = np.random.rand(2)

        def cost_function(beta, gamma):
            return self.square_error(beta, gamma, data)

        residue = [cost_function(beta, gamma)]
        # A few gradient descent steps to improve initial guess:
        stepsize = 0.2
        ctr = 1
        for i in range(20):
            beta, gamma = self.gradient_step(data, stepsize/ctr, beta, gamma)
            residue.append(cost_function(beta, gamma))
            print(cost_function(beta, gamma),', ','Stepsize = ',stepsize/np.sqrt(ctr))
            ctr += 1

        # More gradient descent until delta_new is positive:
        delta_new = -1
        # M=self.hessian(cost_function, [beta, gamma], 0.01)
        while delta_new <= 0 or residue[-1]>5*150/self.N_times:
            adjusted_stepsize = max([0.001,stepsize/ctr])
            beta, gamma = self.gradient_step(data, adjusted_stepsize, beta, gamma)
            r = -self.gradient(cost_function, [beta, gamma], 0.01)
            M = self.hessian(cost_function, [beta, gamma], 0.01)
            s = np.linalg.solve(M, r)
            delta_new = r @ s
            residue.append(cost_function(beta, gamma))
            ctr += 1
            print(residue[-1],', ','Stepsize = ',adjusted_stepsize)
        # Finally, conjugate gradient with improved initial guess:
        print('Starting ConjGrad after',ctr,'steps')
        [beta, gamma], residue_tail = self.conjugate_gradient(cost_function, [beta, gamma])
        residue = residue + residue_tail

        return beta, gamma, residue

    #-------------------------------------------------------------------------------------------------------------------
    # Gradient descent:
    #-------------------------------------------------------------------------------------------------------------------

    def fit_model_gradient(self, data):
        """Computes beta, gamma via gradient descent"""
        # Initialize parameters randomly:
        [beta, gamma] = np.random.rand(2)
        def cost_function(beta, gamma):
            return self.square_error(beta, gamma, data)
        chi = cost_function(beta, gamma)
        residue = [chi]

        # Choose 10 candidates for initial guess; keep the smallest one.
        for i in range(20):
            [b,g] = np.random.rand(2)
            if cost_function(b,g)<chi:
                beta, gamma = b, g
                chi = cost_function(beta, gamma)
                residue.append(chi)

        # Perform gradient descent:
        stepsize = 0.1
        change_in_error = 1
        ctr = 1

        while (change_in_error / chi > 0.1*np.max(np.abs(data)) or chi/self.N_times > 1e-3) and ctr < 1000:
            adjusted_stepsize = stepsize/(np.max([ctr-5,1])+np.exp(ctr/60)-1)
            minimal_stepsize = 0.00005
            if adjusted_stepsize > minimal_stepsize:
                beta, gamma = self.gradient_step(data, adjusted_stepsize, beta, gamma)
            else:
                beta, gamma = self.gradient_step(data, minimal_stepsize, beta, gamma)
            chi2 = cost_function(beta, gamma)
            change_in_error = abs(chi - chi2)
            chi = chi2
            residue.append(chi)
            ctr += 1
            if adjusted_stepsize > minimal_stepsize:
                print('Residual error:',np.round(chi,3), ', Stepsize:',np.round(adjusted_stepsize,5))
            else:
                print('Residual error:', np.round(chi,3), 'Stepsize:', np.round(minimal_stepsize, 5))

        return beta, gamma, residue
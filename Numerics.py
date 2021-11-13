"""
Numerics module for numerical analysis of PDEs - George Whittle 2021
"""

#Import necessary modules
import numpy as np
from scipy.optimize import fsolve

# For testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #Timer for benchmarking
    import time
    start_time = time.time()

"""
Base class for all PDE solvers, contains grid and timestep data,
along with functions to evaluate derivatives, integrate forwards in time etc
"""

class PDE():
    
    def __init__(self, dt = 1e-2, tlim = 20, dx = 1e-2, xlims = np.array([-5, 5])):
        
        # Set model constants
        self.dt = dt        # time increment
        self.tlim = tlim    # Max t to solve up to
        self.dx = dx        # x increment
        self.xlims = xlims  # x limits
        
        # Initialise t vector
        self.t = np.arange(0,tlim,self.dt)
        
        # By default initialise linear x vector
        self.x = self.linearx(False)
        
        # Initialise empty array for system state at all x and t
        self.U = np.zeros([self.x.shape[0],self.t.shape[0]])
        self.U0 = np.zeros([self.x.shape[0]])
        
        # Initialise empty FD differentiation matrix
        self.FDMat = np.zeros([4,3,self.x.shape[0],self.x.shape[0]])
        
    
    """
    Core object functions
    """
    
    def setSystem(self, system: callable):
        
        """
        Used to set the governing differential equation of the PDE instance
        
        Inputs: system - governing PDE of the system, in the form dU/dt = system(U,x,t)
        """

        self.system = system
    
    def setInitConds(self, U0):
        
        """
        Used to set the initial conditions of the PDE instance
        
        Inputs: U0 - either a function of x or a numpy array defined at the instance x coords
        """
        
        if type(U0) == np.ndarray:
            self.U0 = U0
        else:
            self.U0 = np.array([float(U0(a)) for a in np.nditer(self.x)])
        
        self.U[:,0] = self.U0
    
    def solve(self,integrator):
        
        """
        Integrate system forwards in time with integrator to solve to time tlim
        
        Inputs: integrator - integration scheme for time extrapolation
        """
        
        for i in range(self.t.shape[0]-1):
            integrator(i)
    
    def update(self):
        
        """
        Update internal state array shape whenever x or t changes
        """
        
        self.U = np.zeros([self.x.shape[0],self.t.shape[0]])
        
    def setx(self,x):
        
        """
        Set internal x basis to x input
        """
        
        self.x = x
        self.update()


    """
    Various methods to return x vectors which sample at specific points
    """
    
    def linearx(self,updatex = True):
        
        """
        Generate linear x vector across domain according to model parameters
        
        Inputs: updatex - whether to update internal x
        
        Returns: x - array of linear x values
        """
        
        x = np.arange(self.xlims[0],self.xlims[1],self.dx)
        
        if updatex:
            self.x = x
            self.update()
            
        return x
    
    def chebyx(self, N, standard = False, updatex = True):
        
        """
        Generate a vector of the coordinates for the Chebyshev-Gauss-Lobato system
        in the simulation's domain (extrema of Chebyshev polynomials)
        
        Inputs: N - number of collocation points
                standard - whether to transform points to model domain or [-1,1]
                update x - whether to update internal x
        
        Returns: Vector of CGL points
        
        For use with Chebyshev fft differentiator
        """
        
        # Generate x collocation points on [-1,1] (Chebyshev zeros)
        x = np.flip(np.array([np.cos(i * np.pi / N) for i in range(N+1)]))
        
        if not standard:
           # Shift coordinates to match our domain
           x = x * self.xlims.ptp() / 2 + self.xlims.mean()
           
        if updatex:
            # Update internal x
           self.x = x
           self.update()
         
        return x
    
    
    """
    Various methods to compute discrete derivatives
    """    
    
    def finiteDiff(self,f,n=1,o=2):
        
        """
        Calculate the discrete derivative using a minimum order finite differencing scheme
        Based on truncated Taylor series
        Requires uniform x
        
        Inputs: f - the vector containing values of the function to be differentiated
                    evaluated at discrete x
                n - the order of derivative to be calculated, up to 4
                o - the order of leading error, preferably even, up to 6
        
        Returns: df - vector containing nth derivative of f
        """
        
        # Get differentiation matrix, if first off diagonal of stored matrix
        # is 0, it must be built
        if self.FDMat[n-1,(np.ceil(o/2)-1).astype(int),0,1] == 0:
            self.buildFDMat(n,o)
        
        D = self.FDMat[n-1,(np.ceil(o/2)-1).astype(int),:,:]
        
        # Differentiate f
        df = D @ f / self.dx**n
        
        return df
    
    def fftDerivative(self, f, n=1):
        
        """
        Calculate the discrete derivative using discrete Fourier transform
        Based upon trigonometric interpolation
        Requires uniform x
        
        Inputs: f - the vector containing values of the function to be differentiated
                    evaluated at discrete x
                n - the order of derivative to be calculated
                
        Returns: Vector containing nth derivative of f at uniform points
        
        Important - the length of the x vector N must be odd for odd ordered derivatives
                    to correctly be calculated as the Nyquist component cannot be
                    handled well in this case. In the case of even length x vector,
                    for odd derivatives the Nyquist component of the frequency vector
                    will be set to 0, while for even derivatives it will be set
                    to pi * N / L
        
        Warning - aperiodic functions will not be handled perfectly, as while implicit
                  discontinuities at the domain edges have been accounted for in this
                  implementation, there will still be implicit discontinuities in the
                  gradients here as a result leading to Gibb's phenomenon occuring here
        
        Warning - higher order derivatives may exhibit Gibb's phenomenon if a
                  high number of collocation points are used
        """
        
        N=f.shape[0]
        
        # Extend f vector such that it's open periodic (to increase robustness against aperiodicity)
        ftild = np.concatenate((f,np.flip(f[1:-1])))
        
        Ntild = ftild.shape[0]
        
        # Initialize k vector up to Nyquist wavenumber         
        k = np.fft.fftfreq(Ntild,self.dx) * 2 * np.pi
        if int(Ntild/2) == np.ceil(Ntild/2):
            if n % 2 == 1:
                k[int(Ntild/2)] = 0     # As described above
            else:
                k[int(Ntild/2)] *= -1
        
        # Calculate Fourier transformed derivative then inverse transform to obtain the interpolated derivative
        Ftild = np.fft.fft(ftild)
        DFtild = (1j*k)**n * Ftild
        
        dftild = np.real(np.fft.ifft(DFtild))
        df = dftild[0:N]
        return df
    
    def chebyfftDerivative(self, f, n=1):
        
        """
        Calculate the discrete derivative using discrete Fourier transform at
        Chebyshev - Gauss - Lobatto (CGL) points
        Similar to fftDerivative but does not require periodicity
        
        Works as CGL points in x represent uniform points in the variable theta
        
        Inputs: f - function to be differentiated, defined at CGL collocation points
                n - order of derivative, n > 1 will be calculated by recursion
        
        Returns: Vector containing nth derivative of f at CGL collocation points
        
        Works very well for smooth functions, but discontinuities of gradient can
        be very problematic and lead to artfacts at the discontinuities and endpoints
        
        Warning - higher order derivatives may exhibit Gibb's phenomenon if a
        high number of collocation points are used
        """
        
        if n > 1:
            f = self.chebyfftDerivative(f, n-1)
        
        # Extend f vector such that it's defined on theta [0,2pi] not [0,pi]
        ftild = np.concatenate((f,np.flip(f[1:-1])))
        
        # Initialize k vector up to Nyquist wavenumber         
        k = np.fft.fftfreq(ftild.shape[0],np.pi/ftild.shape[0]) *  np.pi
        if int(ftild.shape[0]/2) == np.ceil(ftild.shape[0]/2):
            k[int(ftild.shape[0]/2)] = 0     # As in fftDerivative

        
        # Calculate Fourier transformed derivative wrt theta then inverse transform to obtain the interpolated derivative
        Ftild = np.fft.fft(ftild)
        DFtild = (1j*k) * Ftild
        
        dftild = np.fft.ifft(DFtild)
        
        # Revert to original length
        df = dftild[0:f.shape[0]]
        
        # Use the chain rule to calculate dnf/dx0n where x0 is x on [-1,1]
        x0 = self.chebyx(f.shape[0]-1,standard = True,updatex = False)        
        df[1:-1] /= (np.sqrt(1-x0[1:-1]**2))
        
        # Calculate endpoints limits according to L'Hospital's Rule
        df[[0,-1]] = 0
        
        for i in range(f.shape[0]):
            df[0] += -i**2 * Ftild[i] / f.shape[0]
            df[-1] += (-1)**i * i**2 * Ftild[i] / f.shape[0]
        
        
        # Calculate df/dx
        df /= (self.xlims.ptp() / 2)
        
        return np.real(df)
    
    
    """
    Functions to integrate the system forwards in time
    """
    
    """
    Explicit methods
    """
    
    def RK4(self,i):
        
        """
        Advances4th order Runge - Kutta integration scheme
        
        Inputs: i - index of current time
        
        Returns: Uplus - the next system state
        """
        
        t = self.t[i]
        
        # Evaluate RK4 constants
        K1 = self.dt * self.system(self.U[:,i],self.x,t)
        K2 = self.dt * self.system(self.U[:,i]+K1/2,self.x,t+self.dt/2)
        K3 = self.dt * self.system(self.U[:,i]+K2/2,self.x,t+self.dt/2)
        K4 = self.dt * self.system(self.U[:,i]+K3,self.x,t+self.dt)
        
        # Evaluate dU
        Uplus = self.U[:,i] + (K1 + 2*(K2 + K3) + K4) / 6
        self.U[:,i+1] = Uplus
        
        return Uplus
    
    
    
    """
    Implicit methods
    """
    
    def backEuler(self,i):
        
        """
        Advances system state using Backward Euler Method, simplest possible 
        implicit integration scheme
        First order error
        
        Inputs: i - index of current time
        
        Returns: Uplus - the next system state
        """
        
        t = self.t[i]
        
        eq = lambda dU: self.dt * self.system(self.U[:,i] + dU, self.x, t + self.dt) - dU
        dU = fsolve(eq, np.zeros(self.U[:,i].shape[0]))
        
        Uplus = self.U[:,i] + dU
        self.U[:,i+1] = Uplus
        
        return Uplus
    
    
    """
    Subroutines of other functions
    """
    
    def buildFDMat(self,n,o):
        
        """
        Build finite difference differentiation matrix
        
        Inputs: n - the order of differentiation, up to 4
                o - the order of leading error, even number, up to 6
                
        Implemented such that a given matrix is only built when it's needed
        and is stored for future use
        """
        
        k = (np.ceil(o/2)-1).astype(int)    # Useful integer constant
        
        # Solve coefficient equation for central differencing
        p = (np.floor(0.5*(n+1)) + k).astype(int)
        Arow = np.arange(-p,p+1)
        A = np.zeros([2*p+1,2*p+1])
        for i in range(2*p + 1):
            A[i,:] = Arow**i
            

        b = np.zeros(2*p+1)
        b[n] = np.math.factorial(n)
        
        coeffs = np.linalg.solve(A,b)
        
        # Generate differentiation matrix
        Size = self.FDMat[n-1,k,:,:].shape[0]
        for j in range(coeffs.shape[0]):
            offDiag = j - int((coeffs.shape[0]-1)/2)
            self.FDMat[n-1,k,:,:] += np.diag(coeffs[j]*np.ones(Size-abs(offDiag)),offDiag)
        
        # Solve coefficient equation for forward differencing and pivots up to
        # p positions to account for end derivatives
        N = o + n   # Number of points required
        
        for i in range(p):
            # Solve coefficient equation for ith pivot (0 = forward differencing)
            Arow = np.arange(-i,N-i)
            A = np.zeros([N,N])
            for j in range(N):
                A[j,:] = Arow**j
                
    
            b = np.zeros(N)
            b[n] = np.math.factorial(n)
            
            coeffs = np.linalg.solve(A,b)
            
            # Update left ends
            self.FDMat[n-1,k,i,:N] = coeffs
            
            # Update right ends
            self.FDMat[n-1,k,-i-1,-N:] = (-1)**n * np.flip(coeffs)
        
        
        return self.FDMat[n-1,k,:,:]






# Testing
if __name__ == "__main__":
    heatEq = PDE(dx = 0.1, dt = 0.05)
    alpha = 1
    heatEq.setSystem(lambda U,x,t: alpha*heatEq.finiteDiff(U,2))
    heatEq.setInitConds(lambda x: np.exp(-x**2))
    heatEq.solve(heatEq.backEuler)
    
    #   Generate meshgrid of coordinates
    X, T = np.meshgrid(heatEq.x,heatEq.t)
    
    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, T, heatEq.U.T)




















    print("Execution time:",time.time() - start_time, "seconds")
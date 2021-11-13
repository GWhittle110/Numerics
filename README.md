# Numerics
Numerical Analysis toolkit centred around PDEs, for demonstration and understanding purposes not production

Use procedure:
-  Initialise a new instance of the PDE class, all calculations and variables will be managed by this object

   Inputs: 
-          dt - timestep
-          tlim - time to solve up to
-          dx - x spacing, by default a uniform grid is used
-          xlims - np.array([lower, upper]) bounds of x domain

-  instance.setSystem: Set the governing equation of the PDE in the form dU/dt = lambda U,x,t: some function
-  instance.setInitConds: Set the initial conditions of U in the form lambda x: some function
-  instance.solve: Integrate the system to time tlim using the input integrator (ie RK4, an explicit 4th order Runge Kutta method)


Note on Chebyshev pseudo-spectral differentiator:
- To use this differentiator, the local x coords must be set to the Gauss-Lobatto-Chebyshev collocation points using instance.chebyx(N) where N is the number of points


Known issues:
- 6th order error finite difference matrices can behave badly near domain edges
- fft derivative can cause steady state errors as it implicitly assumes periodic behaviour outside of domain
- Chebyshev differentiator is unstable in current iteration, particularly for higher orders and at the domain boundary, most likely due to Gibb's phenomenon
- More of a use detail, but fft and Chebyshev differentiators rely on compact support. If in doubt, use the finite differencing differentiator


Future implementations:
- Support for boundary conditions
- More integration schemes
- Poentially support for multiple spacial dimensions

function dqdt = controlled_ode(t, x, u, M, D, K)
% controlled_ode: 1-D, first-order, controlled ode of modal dynamics
%   1) Specify system parameters: ode_func = @(t,x) controlled_ode(t, x, u, M, D, K)
%   2) Send to ode solver: [tsol, xsol] = ode45(@ode_func, [0, t_f], x0)
%   
%   Inputs:
%   Current time, t: scalar
%   Current state, x: 2-D vector (v0, q0)
%   Control action, u: matlab function (i.e., u(t) = @(t) interp1(t_span,
%   generalizedForces(ii, :), t) )
%   Mass of mode, M: scalar
%   Damping of mode, D: scalar
%   Stiffness of mode, K: scalar
%
%   Outputs:
%   RHS of first-order ODE, dqdt: Scalar

    % Place holder symbolic variables to be replaced later
    syms q(tau) u_tau

    % Step 1: Form symbolic 1-D, 2nd-order representation of modal dynamics
    second_order_system = M * diff(q, 2, tau) + D * diff(q, tau) + K * q == u_tau;
    [first_order_system, ~] = odeToVectorField(second_order_system);
    
    % Step 2: Substitute in interpolated u at time t
    first_order_system_controlled = subs(first_order_system, u_tau, u(t));
    
    % Step 3: convert controlled first order system into matlab function
    dqdt_func = matlabFunction(first_order_system_controlled, 'vars', {'t', 'Y'});
    dqdt = dqdt_func(t, x);

end
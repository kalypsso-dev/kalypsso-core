# Solving compressible fluid flow equations on AMR grid

## About Euler equations for compressible fluid dynamics

Let's define the vector of conservative variables $U=(\rho, \rho u, \rho v, \rho w, E)$ and the corresponding vector of primitive variables $W=(\rho, u, v, w, p)$, where $\rho$ is the fluid density, $u,v,w$ are the three cartesian components of the velocity vector field, $E$ is the total energy per unit volume, $e$ is the specific internal energy and $p$ is the pressure.

The 1d Euler system of equations written in conservative form reads:

$$
\begin{array}{ccccc}
    \partial_t \rho & + & \partial_x(\rho u) & = & 0,\\
    \partial_t (\rho u) & + & \partial_x(\rho u^2+p) & = & 0,\\
    \partial_t E & + & \partial_x (u(E+p)) & = & 0,\\
  \end{array}
$$

or in short notations:

$$
\partial_t \mathbf{U} + \partial_x \mathbf{F(U)} = \mathbf{0}
$$

where the flux function is defined by

$$
\mathbf{F(U)} = \left [
  \begin{array}{c}
    \rho u \\
    \rho u^2 + p \\
    u (E + p)
  \end{array} \right]
$$

To close the system, one needs an additional relation, often given by an equation of state (EoS), that is a relation $e=e(\rho,p)$; that is internal energy is a function of density and pressure.

In case of ideal gas EoS, one has

- total (internal + kinetic) energy per unit volume $E = \rho \left( e + \frac{1}{2} u^2 \right) = \frac{p}{\gamma-1} + \frac{1}{2} \rho u^2$,
- specific (per mass unit) internal energy $e=\frac{p}{(\gamma-1)\rho}$.
- total enthalpy $H = (E + p)/\rho$ per mass unit (specific enthalpy)
- another useful relation $\frac{\gamma E}{\rho}=H+(\gamma-1)u^2/2$.
- Let $c=\sqrt{\left(\frac{\partial p}{\partial \rho}\right)_s}$ be the speed of sound. For ideal gas EoS, one can show that $c = \sqrt{\frac{\gamma p}{\rho}}$.

## Euler equations in primitive variables

Let's do a change of variable, and define $W=(\rho, u, \rho e)$.

One aims at rewriting 1D Euler equations in quasi-linear form:

$$
    Q_t + A(Q) Q_x = 0
$$

before, doing that, lets rewritten 1D Euler equation without any assumption about EoS:

$$
\begin{array}{ccc}
    \partial_t \rho + u \partial_x \rho + \rho \partial_x u & = & 0\\
    \partial_t u + u \partial_x u + \frac{1}{\rho} \partial_x p & = & 0\\
    \partial_t(\rho e) + u \partial_x(\rho e) + \rho e \partial_x u + P \partial_x u & = & 0
\end{array}
$$

If one assumes ideal gas EoS, internal energy can be computed exactly as $\rho e = \frac{p}{\gamma-1}$, one can obtain the quasi-linear form above, where

$$
A(Q) = \left[
\begin{array}{ccc}
u & \rho & 0\\
0 & u    & 1/rho\\
0 & \rho c^2 & u
\end{array} \right]
$$

If one assumes stiffened gas Eos, the definition of $A(Q)$ still holds, we just need to adjust the definition of speed of sound $c$.

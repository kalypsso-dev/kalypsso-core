# Notes about the Van der Waals equation of state

The extensive form of the [van der Waals EOS](https://en.wikipedia.org/wiki/Van_der_Waals_equation) reads

```math
(P + \frac{an^2}{V^2})(V - n b ) = n R T
```

or

```math
P = \frac{n R T}{V - n b} - \frac{a n^2}{V^2}
```


where

- $`P`$ is thermal pressure,
- $`V`$ is volume,
- $`T`$ is temperature,
- $`R`$ is the universal perfect gas constant,
- $`n`$ is the amount of substance (in mole),
- $`a`$ is cohesion term,
- $`b`$ is molar covolume.

## internal energy as a function of pressure and density

In order to use it in kalypsso, we need to derive the internal energy (either specific or volumic).

Let's start by computing partial derivative of internal energy with respect to volume at constant temperature : $`\frac{\partial  U}{\partial V}\bigg|_T`$.

From thermodynamics first principle applied to a gas system: $`dU=T dS -P dV`$ or in term of free energy  $`dF=-S dT -P dV`$. So that

```math
\frac{\partial  U}{\partial V}\bigg|_T = \frac{\partial (F+TS)}{\partial V}\bigg|_T
=  \frac{\partial F}{\partial V}\bigg|_T + T  \frac{\partial S}{\partial V}\bigg|_T
= -P + T  \frac{\partial P}{\partial T}\bigg|_V
= T^2 \frac{\partial (P/T)}{\partial T}\bigg|_V
```

We can integrate the last equation using a reference state:
```math
U(T,V,N) = U_{ref}(T,V_{ref},N) + \int_{V_{ref}}^V  T^2 \frac{\partial (P/T)}{\partial T}\bigg|_V dV
```

When volume goes to infinity, the gas behaves as a ideal gas, so

```math
U(T,V,N) = U_{ideal}(T) +  \int_{\infty}^V  T^2 \frac{\partial (P/T)}{\partial T}\bigg|_V dV = n C_v T - \frac{a n^2}{V} = m c_v T - \frac{a n^2}{V}
```
where $`c_v`$ is the specific heat capacity at constant volume. Using the definition of $`gamma`$ the specific heat ratio, one obtains $`c_v=\frac{R}{(\gamma-1)M}`$

Finally, one obtains the relation between pressure and volumic internal energy $`\rho e = \frac{U}{V}`$:

```math
P = \frac{\gamma-1}{1-\rho \frac{b}{M}} (\rho e + \frac{a \rho^2}{M}) - \frac{a \rho^2}{M}
```

where:

- $`M`$ is molar mass
- $`\rho`$ is volumic mass

By renormalizing $`a`$ and $`b`$, one finally obtains (see equation (6) in [Numerical Simulations of Compressible Two-Component Flows with General Equation of State](https://www.researchgate.net/publication/381769124_NUMERICAL_SIMULATIONS_OF_COMPRESSIBLE_TWO-_COMPONENT_FLOWS_WITH_GENERAL_EQUATION_OF_STATE) )

```math
P = \frac{\gamma-1}{1-\rho b} (\rho e + a \rho^2) - a \rho^2
```

## Speed of sound

By definition of speed of sound $`c=\sqrt{\frac{\partial P}{\partial \rho}\bigg_S}`$.
In order to derive the actual expression pf $`c`$, one uses alternate expression (see book by Toro, equation 1.36):

```math
c^2=\frac{\frac{P}{\rho^2}-\frac{\partial e}{\partial \rho}\bigg_P}{\frac{\partial e}{\partial P}\bigg_\rho}
```

For the van der Waals EOS, one obtains:
```mat
c^2=\frac{\gamma P + (\gamma-2) a \rho^2 + 2 a b \rho^3}{(1 - b \rho)\rho}
```

We obtains of course the same expression as the one given in Pantano et al (see below), but more straightforwardly.

## References about van der Waals EOS

- Thermokinetic model of compressible multiphase ﬂows, Ehsan Reyhanian, Benedikt Dorschner, Ilya Karlin. https://arxiv.org/abs/2002.09217; see equation A.17 for speed of sound
- An oscillation free shock-capturing method for compressible van der Waals supercritical fluid flows, C. Pantano , R. Saurel, T. Schmitt. Journal of Computational Physics,
Volume 335, 2017, Pages 780-811, https://doi.org/10.1016/j.jcp.2017.01.057; see equation A.26 for speed of sound (after some rewriting it is exactly the same as previous reference)

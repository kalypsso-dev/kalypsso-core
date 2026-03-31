# Notes about the Mie-Gruneisen EOS

The Mie-Gruneisen EOS is a general form of EOS, written as:

```math
P(\rho,e) = \rho\Gamma(\rho) \left[ e - e_{\text{ref}}(\rho) \right] + P_{\text{ref}}(\rho),
```

where $`e`$ is specific internal energy and $`\Gamma`$ is the so-called Gruneisen parameter, thermodynamically defined as

```math
\Gamma(\rho) = \frac{1}{\rho} \frac{\partial P}{\partial e}\bigg\rvert_{1/\rho}
```

$`P_{\text{ref}}(\rho)`$ and $`e_{\text{ref}}(\rho)`$ are reference states which can be taken along isentropic, isothermal or shock Hugoniot compression curves.

By carefully choosing the reference pressure and reference internal energy, many different EOS can be put into Mie-Gruneisen form ([M.A. Price et al, A method for compressible multimaterial flows with condensed phase explosive detonation and airblast on unstructured grids](https://doi.org/10.1016/j.compfluid.2015.01.006)):

| | Common form | $`\Gamma(\rho)`$ | $`p_{\text{ref}}(\rho)`$ | $`e_{\text{ref}}(\rho)`$|
|---|-------------|------------------|--------------------------|-------------------------|
| Ideal gas | $`P=(\gamma-1)\rho e`$ | $`\gamma-1`$ | 0 | 0 |
| Stiffened gas | $`P=(\gamma-1)\rho e-\gamma a`$ | $`\gamma-1`$ | $`-\gamma a`$ | 0 |
| Tail | $`P=(\gamma-1)\rho e-\gamma (b-a)`$ | $`\gamma-1`$ | $`-\gamma (b-a)`$ | 0 |
| van der Waals | $`P=\frac{\gamma-1}{1-b\rho}(\rho e+a\rho^2)-a\rho^2`$ | $`\frac{\gamma-1}{1-b\rho}`$ | $`-a\rho^2`$ | $`-a\rho`$ |
| | | | | |
| Cochran-Chan |  | $`\Gamma_0`$ | $`AV^{-\epsilon_1}-BV^{-\epsilon_2}`$ | $`\frac{-A}{\rho_0(1-\epsilon_1)}\left[ V^{1-\epsilon_1}-1\right] + \frac{B}{\rho_0(1-\epsilon_2)}\left[ V^{1-\epsilon_2}-1\right]-e_0`$ |
| JWL | | $`\omega`$ | $`A e^{-R_1 V} + B e^{-R_2 V}`$ | $`\frac{A}{\rho_0 R_1} e^{-R_1 V} + \frac{B}{\rho_0 R_2} e^{-R_2 V} - e_0`$ |
| Shock-wave | | $`\Gamma_0 (\frac{\rho_0}{\rho})^\alpha`$ | $`p_0+\frac{c_0^2 (1/\rho_0-1/\rho)}{\left[ 1/\rho - s(1/\rho_0-1/\rho)\right]^2}`$ | $`e_0+\frac{1}{2}\left[ P_{\text{ref}}(\rho)+P_0\right](\frac{1}{\rho_0}-\frac{1}{\rho})`$ |


## Sound speed

Sound speed for a general EOS is defined as (see [Price et al.](https://doi.org/10.1016/j.compfluid.2015.01.006))):

```math
c^2 = \left( \frac{\partial P}{\partial \rho}\right)_S = \left( \frac{\partial P}{\partial \rho}\right)_e + \frac{P}{\rho^2} \left( \frac{\partial P}{\partial e}\right)_\rho
```

For an EOS in Mie-Gruneisen form, one obtains:

```math
c^2 = \frac{d P_{\text{ref}}}{d\rho} -\rho\Gamma \frac{d e_{\text{ref}}}{d\rho} \frac{\Gamma P}{\rho} + (P-P_{\text{ref}})*\left[\frac{1}{\rho} +\frac{1}{\Gamma}\frac{d \Gamma}{d \rho} \right]
```

## Isentropic bulk modulus

Using internal code, Mie-Gruneisen EOS is $`P=\phi(\rho)+\rho\Gamma(\rho)(e_{\text{int}}-e_H)`$ where $`e_H=e_{\text{ref}}`$ is the reference internal energy taken along a Hugoniot curve.

Let's first compute $`\frac{\partial P}{\partial \rho}\bigg\rvert_{e_{\text{int}}}`$:

```math
\frac{\partial P}{\partial \rho}\bigg\rvert_{e_{\text{int}}} = \frac{\partial \phi}{\partial \rho}\bigg\rvert_{e_{\text{int}}} + \frac{d\Gamma}{d\rho}\rho(e_{\text{int}}-e_H) + \Gamma (e_{\text{int}}-e_H) - \Gamma \rho \frac{de_H}{d\rho}
```

In the compressive case, isentropic bulk modulus is $`\kappa = \rho\frac{d P}{d\rho}|_{e_{\text{int}}} + \frac{P}{\rho} \frac{d P}{d e_{\text{int}}}|_{\rho}`$

TODO: why not use `$\kappa = \rho c^2$` (it should be equal to previous relation) ?

# Bibliographic references

- [M.A. Price et al, A method for compressible multimaterial flows with condensed phase explosive detonation and airblast on unstructured grids](https://doi.org/10.1016/j.compfluid.2015.01.006)
- [G.H. Miller and E.G. Puckett, A high-order Godunov method for multiple condensed phases, Journal of Computational Physics, Volume 128, Issue 1, 1 October 1996, Pages 134-164](https://doi.org/10.1006/jcph.1996.0200)
- [R.M. Cheng et al., GnarlyX: Eulerian multi-material hydrodynamics coupled to to equation of state and hyperelastic, plastic constitutive models.](https://www.osti.gov/biblio/1963620)
- [S.C. Dick and E. Johnsen, An analytical method for determining stiffend equation of state parameters from shock-compression experiments, Physica D: Nonlinear Phenomena, Volume 490, 2026, 135179](https://doi.org/10.1016/j.physd.2026.135179)
- [Allen C. Robinson, The Mie-Gruneisen Power Equation of State, Sandia Report SAND2019-6025](https://www.osti.gov/servlets/purl/1762624)
- [Mie–Grüneisen equation of state (wikipedia)](https://en.wikipedia.org/wiki/Mie%E2%80%93Gr%C3%BCneisen_equation_of_state)


Software:

- [singularity-eos](https://github.com/lanl/singularity-eos)

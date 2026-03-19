#  SPDX-FileCopyrightText: 2025 kalypsso contributors
#
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- coding: utf-8 -*-

import numpy as np

class StiffenedGas:

    def __init__(self, gamma0, gamma1, pinf0, pinf1):
        self.setup(gamma0, gamma1, pinf0, pinf1)

    def __compute_epsilon(self, phi0):
        return phi0 / (self.gamma0 - 1) + (1 - phi0) / (self.gamma1 - 1)

    def __compute_gamma_pinf(self, phi0):
        epsilon = self.__compute_epsilon(phi0)
        phi1 = 1.0 - phi0
        return ((phi0 * self.pinf0 * self.gamma0) / (self.gamma0 - 1.0) +
                (phi1 * self.pinf1 * self.gamma1) / (self.gamma1 - 1.0)) / epsilon

    def mixture_pressure(self, rho, eint, phi0):
        small_p = 1e-6
        epsilon = self.__compute_epsilon(phi0)
        gamma_pinf = self.__compute_gamma_pinf(phi0)
        pressure = rho * eint / epsilon - gamma_pinf
        return small_p if pressure < small_p else pressure

    def mixture_specific_eint(self, pressure, rho, phi0):
        epsilon = self.__compute_epsilon(phi0)
        gamma_pinf = self.__compute_gamma_pinf(phi0)
        return (pressure + gamma_pinf) * epsilon / rho

    def mixture_sound_speed(self, pressure, rho, phi0):
        epsilon = self.compute_epsilon(phi0)
        gamma_pinf = self.compute_gamma_pinf(phi0)
        return sqrt(1 / rho * (gamma_pinf + pressure * (1 + 1 / epsilon)))

    def setup(self, gamma0, gamma1, pinf0, pinf1):
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.pinf0 = pinf0
        self.pinf1 = pinf1


###############################################################################
if __name__ == "__main__":
    # water
    gamma0=4.4
    pinf0=6e8

    # air
    gamma1=1.4
    pinf1=0.0

    sg = StiffenedGas(gamma0, gamma1, pinf0, pinf1)

    print("stiffened gas pressure {}".format(sg.mixture_pressure(1000.0, 1e8, 0.9)))

# SPDX-FileCopyrightText: 2025 kalypsso-core authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np

import sympy as sym
from fractions import Fraction
import math

sym.init_printing()

x = sym.symbols('x')
a,b,c = sym.symbols('a b c')
h = sym.symbols('h')


#
# integrate 2nd order polynomial ax^2+bx+c
#

P = (a*x**2+b*x+c)/h

# coarse cells (left, center and right)
sym.integrate(P, (x,-3*h/2, -h/2))
sym.integrate(P, (x,  -h/2,  h/2))
sym.integrate(P, (x,   h/2,3*h/2))

m = sym.Matrix([[13/12*h**2, -h, 1],
                [ 1/12*h**2,  0, 1],
                [13/12*h**2,  h, 1]])
m

# inverse matrix to get a,b,c
m_inv=m.inv()
m_inv
Fraction(-0.0416666666666667).limit_denominator(1_000_000)
Fraction(1.08333333333333).limit_denominator(1_000_000)

# there are the conservative values in left, center and right cells
Im1, I0, Ip1 = sym.symbols('Im1 I0 Ip1')
I = sym.Matrix([Im1,I0,Ip1])

# compute a,b,c
I2=m_inv*I

aa=I2[0]
bb=I2[1]
cc=I2[2]

P2 = (aa*x**2+bb*x+cc)/(h/2)

# integrate over child cells to get fine prolongated values
sym.integrate(P2, (x,  -h/2,  0  ))
sym.integrate(P2, (x,     0,  h/2))

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
a,b,c,d,e = sym.symbols('a b c d e')
h = sym.symbols('h')


#
# integrate 4th order polynomial ax^4+bx^3+cx^2+dx+e
#

P = (a*x**4 + b*x**3 + c*x**2 + d*x + e)/h

# coarse cells (2 on the left, center and two on the right)
sym.integrate(P, (x,-5*h/2, -3*h/2))
sym.integrate(P, (x,-3*h/2,   -h/2))
sym.integrate(P, (x,  -h/2,    h/2))
sym.integrate(P, (x,   h/2,  3*h/2))
sym.integrate(P, (x, 3*h/2,  5*h/2))

m = sym.Matrix([[1441/80*h**4, -17/2*h**3, 49/12*h**2, -2*h, 1],
                [ 121/80*h**4,  -5/4*h**3, 13/12*h**2,   -h, 1],
                [   1/80*h**4,          0,  1/12*h**2,    0, 1],
                [ 121/80*h**4,   5/4*h**3, 13/12*h**2,    h, 1],
                [1441/80*h**4,  17/2*h**3, 49/12*h**2,  2*h, 1]])
m

# inverse matrix to get a,b,c,d,e
m_inv=m.inv()
m_inv
Fraction(-0.0416666666666667).limit_denominator(1_000_000)
Fraction(1.08333333333333).limit_denominator(1_000_000)

# there are the conservative values in left, center and right cells
Im2, Im1, I0, Ip1, Ip2 = sym.symbols('Im2 Im1 I0 Ip1 Ip2')
I = sym.Matrix([Im2,Im1,I0,Ip1,Ip2])

# compute a,b,c,d,e
I2=m_inv*I

aa=I2[0]
bb=I2[1]
cc=I2[2]
dd=I2[3]
ee=I2[4]

P2 = (aa*x**4 + bb*x**3 + cc*x**2 + dd*x + ee)/(h/2)

# integrate over child cells to get fine prolongated values
sym.simplify(sym.integrate(P2, (x,  -h/2,  0  )))
sym.simplify(sym.integrate(P2, (x,     0,  h/2)))

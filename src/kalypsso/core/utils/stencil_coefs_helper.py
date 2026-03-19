#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 kalypsso-core authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np

from fractions import Fraction
import math

def cross_check(x0, delta_h, order, offsets, coeffs):
    ''' cross-check finite difference approximation
    '''

    # define a function f and its 1st and 2nd derivative
    def f(x):
        return 7*x**7 + 2*x**3 - 6*x + 1
    def d1f(x):
        return 49*x**6 + 6*x**2 - 6
    def d2f(x):
        return 294*x**5 + 12*x
    def d3f(x):
        return 1470*x**4 + 12
    def d4f(x):
        return 5880*x**3
    def d5f(x):
        return 17640*x**2
    def d6f(x):
        return 35280*x
    def d7f(x):
        return 35280

    def eval_finite_diff(x, dh):
        value = 0.0
        for i in range(coeffs.size):
            value += coeffs[i]*(f(x+offsets[i]*dh)-f(x))/(dh**order)
        return value

    # compare exact derivative at x0 with finite difference approximation
    if order == 1:
        exact_value = d1f(x0)
    elif order == 2:
        exact_value = d2f(x0)
    elif order == 3:
        exact_value = d3f(x0)
    elif order == 4:
        exact_value = d4f(x0)
    else:
        print("Warning checking only available up to order 4")
        return

    approx_value = eval_finite_diff(x0, delta_h)
    abs_error = np.abs(exact_value-approx_value)
    rel_error = abs_error/exact_value
    normalized_error = (exact_value-approx_value)/delta_h**(coeffs.size)

    print("exact derivative : {}".format(exact_value))
    print("approximate value: {}".format(approx_value))
    print("absolute error   : {}".format(abs_error))
    print("relative error   : {}".format(rel_error))
    print("normalized error : {}".format(normalized_error))

def compute_finite_difference_coefficients(order, offsets, x0, delta_h):
    '''Compute finite difference coefficients needed to estimate a first or second order derivative.
    Coefficients are computed by solving a linear system obtained from the Taylor expansion about the given points.

    Taking as example the finite difference estimation of the first order derivative, using three points located at [alpha, 0, beta]. Doing the Taylor expansion about 0 (central point) of function f up to order 2:
    f(alpha) = f(0) + alpha f'(0) + alpha^2/2 f''(0) + ...
    f(beta) = f(0) + beta f'(0) + beta^2/2 f''(0) + ...

    to estimate f'(0) we need to find coefficients a and b such that:
    a *  alpha + b * beta = 1
    a * alpha^2/2 + b * beta^2/2 = 0

    We use a linear system solver to obtain coefficient a and b
    '''

    # solve M * X = Y
    M = np.array([offsets**i for i in range(1,offsets.size+1)])
    Y = np.zeros(offsets.size)
    Y[order-1] = math.factorial(order)

    X = np.linalg.solve(M, Y)

    print("Given offsets {}, finite difference coefficients to estimate order {} derivative are {}".format(offsets,order,X))
    print("Trying to get fractions:")
    for i in range(X.size):
        print("coeff[{}]={}={}".format(i,X[i],Fraction(X[i]).limit_denominator(1_000_000)))

    cross_check(x0, delta_h, order, offsets, X)


###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute finite difference coefficients.',
                                     epilog="Example: %(prog)s -o 1 -p -1.0 1.0")
    parser.add_argument('-o', '--order', metavar="order", type=int, default=1)
    parser.add_argument('-p', '--pos', metavar="pos", type=float, nargs='+', help="Points relative positions to center point")
    parser.add_argument('-x', metavar="x0", type=float, default=2.0, help="Abscissa of the point where derivatie is estimated")
    parser.add_argument('-d', metavar="delta_h", type=float, default=0.01, help="Cell size of central point")
    parser.add_argument('-f', default=False, action="store_true", help="print full help")
    args = parser.parse_args()

    if args.f:
        parser.print_help()
        print(compute_finite_difference_coefficients.__doc__)

    else:
        # check that pos array contains at least order values
        if not len(args.pos) >= args.order:
            print("Error option -p / --pos should have at least \"order\" arguments!\n")
            parser.print_help()
            exit()

        compute_finite_difference_coefficients(args.order, np.array(args.pos), args.x, args.d)

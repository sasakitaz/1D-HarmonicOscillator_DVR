# 1D-HarmonicOscillator_DVR.py

## What is it?
This program solve Schrodinger equation on 1-dimensional Morse oscillator by harmonic oscillator DVR basis.

## Usage
You can change parameters at the top of the program and solve Schrodinger equation on 1-dimensional Morse oscillator.

The program compares numerical results on a grid basis, a harmonic oscillator basis, and analytical solutions.

parameters

1. n0: matrix size
2. m: mass (a.m.u.)
3. kzz: Morse potential parameter
4. az: Morse potential parameter

## Output
1. Ri eigen value: Gauss quadrature points for grid basis
2. DVR basis eigen value: the eigenvalue of 1-dimensional Morse oscillator by using grid basis
3. HO basis eigen value: the eigenvalue of 1-dimensional Morse oscillator by using harmonic oscillator basis (spectral basis)
4. Figure: the potential surface solved by the program and Gauss quadrature points

# QuAVL

## Specification-based Verification and Repair of Quantum Circuits

This Python project comes with the experiments

- Quantum Teleportation Protocol
- Deutsch-Jozsa Algorithm
- Quantum Fourier Transform
- Quantum implementations of classical algorithms (RevLib)

that show how QuAVL can prove correctness of well-known quantum algorithm.

Further QuAVL can synthesize repaired quantum programs in case of faults
such as changed phase or omitted gates.


## Installation

There are three requirements for QuAVL to run. The Python packages are:

- numpy (for arithmetic)
- pyparsing (needed for output parsing)
- z3-solver (for SMT sort declaration and syntax generation)

Also, the system QuAVL runs on requires [dReal](http://dreal.github.io)
to be installed and added to the path in order for the solver to work.
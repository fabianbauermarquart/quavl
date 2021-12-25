from typing import List

import numpy as np
from z3 import If, Or, Not

from quavl.lib.constants import cos, sin, pi
from quavl.lib.expressions.complex import ComplexVal
from quavl.lib.expressions.qbit import Qbits, QbitVal
from quavl.lib.models.circuit import Circuit, Method
from quavl.lib.operations.gates import H, R, SWAP, Rx, Rz
from quavl.lib.solver import SpecificationType


def create_qft_circuit(n: int, swap: bool = False):
    """
    Create QFT circuit.
    :param n: number of qbits.
    :param swap: true if output qbits should be swapped.
    :return: circuit.
    """
    qbits, program = create_qft_instructions(n, swap)

    return Circuit(qbits, program)


def create_qft_instructions(n: int, swap: bool = False):
    """
    Create QFT instructions.
    :param n: number of qbits.
    :param swap: true if output qbits should be swapped.
    :return: Tuple of qbits and program.
    """
    qbits = Qbits([f'q{i}' for i in range(n)])

    program = []

    for i in range(n):
        program.append(H(qbits[i]))

        for k in range(2, n - i + 1):
            program.append(R(qbits[i], k).controlled_by(qbits[i + k - 1]))

    if swap:
        for i in range(int(np.floor(n / 2))):
            program.append(SWAP(qbits[i], qbits[n - (i + 1)]))

    return qbits, program


def build_qft_specification(qbits: List[QbitVal]):
    """
    Build specification for QFT.
    :param qbits: qbits.
    :return: specification.
    """
    n = len(qbits)

    output_bit_values = [If(q.beta.r == 1.0, 1.0, 0.0) for q in qbits]

    output_bit_fractions = []

    for i in range(n):
        output_bit_fraction = 0.0

        for j, k in enumerate(range(i, n)):
            output_bit_fraction += output_bit_values[k] * (1 / (2 ** (j + 1)))

        output_bit_fractions.append(output_bit_fraction)

    inv_root2 = 1 / np.sqrt(2)  # inv_root2 = 1 / Sqrt(Real(2))

    output_specification = [QbitVal(
        alpha=ComplexVal(r=inv_root2),
        beta=ComplexVal(r=inv_root2 * cos(2 * pi * v),
                        i=inv_root2 * sin(2 * pi * v)))
        for v in output_bit_fractions]

    return output_specification


def prove_qft(n: int):
    # Build circuit
    circuit = create_qft_circuit(n)

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize_qbits(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    for i in range(n):
        circuit.solver.add(spec_qbits[i] == output_specification[i])

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_smt_encoding=False)


def repair_qft(n: int):
    # Build circuit
    circuit = create_qft_circuit(n)

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize_qbits(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    for i in range(n):
        circuit.solver.add(spec_qbits[i] == output_specification[i])

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_smt_encoding=False,
                  synthesize_repair=True)


def verify_repaired_qft(n: int):
    # Build circuit
    qbits, program = create_qft_instructions(n)

    rep_phi = [-0.0493] * len(qbits)
    rep_theta = [0.0493] * len(qbits)

    for i in range(len(qbits)):
        program.append(Rx(qbits[i], rep_theta[i]))
        program.append(Rz(qbits[i], rep_phi[i]))

    circuit = Circuit(qbits, program)

    initial_values = [{(1, 0), (0, 1)} for _ in range(n)]

    circuit.initialize_qbits(initial_values)

    # Build specification
    final_qbits = circuit.get_final_qbits()
    spec_qbits = Qbits([f'spec_q{i}' for i in range(n)])

    output_specification = build_qft_specification(circuit.qbits)

    disjunction = []

    for i in range(n):
        disjunction.append(Not(spec_qbits[i] == output_specification[i]))

    circuit.solver.add(Or(disjunction))

    circuit.final_qbits += spec_qbits

    # Set specification
    circuit.set_specification([(final_qbits[i], spec_qbits[i]) for i in range(n)],
                              SpecificationType.equality_pair_list)

    # Prove
    circuit.prove(method=Method.qbit_sequence_model,
                  dump_smt_encoding=False)


if __name__ == "__main__":
    prove_qft(4)
    repair_qft(4)
    verify_repaired_qft(4)

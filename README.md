# Quantum Computing for Genomics

## Phase 0: Foundational Math (Week 1)
- Complex numbers and linear algebra for QC
  - Matrix multiplication (NumPy)
  - Eigenvalues/vectors (for Hamiltonians)
  - Dirac notation (kets, bras, inner products)
- Probability basics
  - Measuring quantum states (Born rule)
  - Stochastic vs quantum randomness

## Phase 1: Quantum Programming Basics (Weeks 2-3)
### Qubits and Gates
- Quantum states vs classical bits
- Single-qubit gates (X, Y, Z, H) with Qiskit
- Visualization: Bloch sphere plots
- Hands-on: Simulate state rotations in Python

### Multi-Qubit Systems
- Entanglement and Bell states
- Two-qubit gates (CNOT, SWAP)
- Measuring multi-qubit systems
- Lab: Entanglement detection with Qiskit

## Phase 2: Genomic Applications
### DNA Encoding
- Binary vs amplitude encoding of sequences
- Mapping ACTG to qubit states
- Challenge: Store 4-base sequence in 3 qubits
- Lab: Sequence alignment oracle

### Grover's Algorithm for Genomics
- Pattern matching in DNA sequences
- Oracle design for motif search
- Speedup analysis vs BLAST
- Project: Find "TATA" box in simulated DNA

### Variational Algorithms
- Quantum Approximate Optimization (QAOA)
  - For genome assembly problems
- Variational Quantum Eigensolver (VQE)
  - Protein folding energy minimization
- Lab: Small peptide folding simulation

## Phase 3: Advanced Topics
### Error Mitigation
- Noise models in genomic circuits
- Readout error correction
- Zero-noise extrapolation
- Lab: Error-aware sequence alignment

### Hybrid Algorithms
- Classical pre-processing + quantum core
- Quantum machine learning for SNP detection
- Project: Hybrid variant caller

## Tools and Libraries
- Qiskit (IBM)
- PennyLane (Xanadu)
- Cirq (Google)
- Q# (Microsoft) for genomics-specific packages


## Resources
- Qiskit Textbook (free online)
- "Quantum Algorithm Implementations for Beginners" (arXiv)
- Rosalind.info problems (quantum adaptation)

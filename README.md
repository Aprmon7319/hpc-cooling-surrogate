# HPC Cooling Surrogate

**Deep Learning Surrogate Models for FMU-Based Cooling System Simulation in HPC Digital Twins**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

This repository provides deep learning surrogate models that replace computationally expensive FMU-based cooling-system simulators in the [ExaDigiT](https://exadigit.readthedocs.io/) digital twin framework for high-performance computing (HPC) data centers.

The **Federated DeepM&Mnet** architecture achieves:
- 📊 **Mean R² = 0.963** across 2,827 output variables (97.2% exceeding R² > 0.8)
- ⚡ **20,710× speedup** over FMU simulation (3.16 ms vs. 65.43 s per prediction)
- 🔄 **2.84M× peak throughput** with batched GPU inference

This enables real-time cooling optimization, rapid what-if scenario analysis, and Monte Carlo uncertainty studies that would otherwise be computationally prohibitive.

---

## Key Features

- **Modular Architecture**: Shared temporal encoder with six specialized decoder heads for heterogeneous output dynamics
- **Multi-System Support**: Pre-trained models for Summit (257 CDUs), Marconi100 (49 CDUs), and Lassen (44 CDUs) supercomputer configurations
- **Comprehensive Data Generation**: Realistic operational scenarios and systematic exploration campaigns
- **Physics-Informed Option**: Differentiable physics loss framework (though empirically shown to be redundant for high-fidelity FMU data)
- **Extensible Design**: Easily adaptable to new configurations via ExaDigiT's AutoCSM module

---

## Useful Information

- **Data Link**: https://doi.org/10.5281/zenodo.19595930 

# LimitRegressor  
**Converging is in fashion**

Author: Eduard Arzumanyan  
Date: December 28, 2025  

---

## 📌 Overview

**LimitRegressor** is a regression model built around the idea of convergence.  
Instead of producing predictions in a single forward pass, the model refines its output iteratively until it stabilizes.

The document presents the full theoretical foundation, training methodology, convergence guarantees, complexity analysis, and pseudocode implementation of the model.

This repository contains the formal description of the architecture and its mathematical justification.

---

## 📖 What This Document Contains

The paper is structured as follows:

### 1️⃣ Abstract
A high-level summary of the model, its motivation, and its main contributions:
- Fixed-point based regression
- Guaranteed convergence under mild conditions
- Efficient gradient computation
- Constant-memory training
- Complexity analysis

---

### 2️⃣ Introduction
Explains:
- Why LimitRegressor is different from standard feed-forward models
- How it relates to RNNs and Deep Equilibrium Models
- The intuition behind iterative refinement
- Basic mathematical definitions required to understand the model

This section also introduces fundamental calculus concepts needed for the theoretical proofs later in the paper.

---

### 3️⃣ Related Work
Discusses:
- Iterative refinement models
- Fixed-point iteration
- Lipschitz continuity
- Contraction mappings
- Theoretical background ensuring convergence and uniqueness

This section builds intuition and theoretical grounding for the proposed approach.

---

### 4️⃣ The LimitRegressor Model
The core of the paper.

It defines:
- The iterative prediction mechanism
- Model parameters and hyperparameters
- Convergence conditions
- The prediction process

It also introduces the key idea of differentiating through a fixed point instead of unrolling all iterations during training.

Subsections include:
- Differentiation through the fixed point
- Convergence of unrolled gradients
- Training algorithm
- Prediction algorithm

---

### 5️⃣ Complexity Analysis
A detailed computational analysis covering:
- Inference complexity
- Training complexity
- Memory requirements
- Worst-case bounds
- Logarithmic convergence behavior

This section demonstrates that training memory does not grow with the number of refinement iterations.

---

### 6️⃣ Pseudocode
Clear algorithmic descriptions of:

- Prediction via fixed-point iteration
- Training via implicit differentiation
- Baseline training via unrolled backpropagation

This makes the model directly implementable.

---

## 🚀 Key Characteristics of LimitRegressor

- Iterative prediction refinement
- Convergence-based output definition
- Unique fixed-point solution
- Adjustable accuracy vs computation trade-off
- Implicit gradient computation
- Constant memory training
- Theoretical convergence guarantees

---

## 🎯 Who Is This For?

This document is intended for:

- Machine learning researchers
- Students interested in theoretical ML
- People studying fixed-point methods
- Anyone exploring equilibrium-based models
- Developers interested in memory-efficient training methods

---

## 📌 Implementation Notes

The paper provides:
- Full theoretical justification
- Algorithmic structure
- Clear stopping criteria
- Training and prediction workflows

An implementation can be built in any deep learning framework (PyTorch, TensorFlow, JAX, etc.) using the described procedures.

---

## 📜 Citation

If you reference this work, please cite:

Eduard Arzumanyan.  
*LimitRegressor: Converging is in fashion.*  
December 28, 2025.

---

## ✨ Final Note

LimitRegressor is not just another regression model.  
It is a convergence-driven framework that blends calculus, fixed-point theory, and machine learning into a structured and theoretically grounded system.

# MA_AdaLoRA: Magnitude-Aware AdaLoRA

**NLP Course Project Code**

## Introduction

This repository hosts the implementation of **Magnitude-Aware AdaLoRA**, a robust parameter-efficient fine-tuning (PEFT) method developed for a Natural Language Processing course project.

Standard AdaLoRA relies heavily on gradient sensitivity for rank allocation, which can be unstable in low-resource or high-noise scenarios (e.g., small batch sizes). This project improves upon the original AdaLoRA by introducing a **Magnitude-Aware** mechanism that incorporates static weight magnitude as a stable prior. This "dual-stream" importance evaluation (Gradient + Magnitude) significantly mitigates catastrophic pruning and enhances model robustness on tasks like CoLA.

## Features

* **Magnitude-Aware Regularization**: Combines dynamic gradient sensitivity with static weight magnitude for importance scoring.
* **Robust Rank Allocation**: Prevents the accidental pruning of critical singular values during the warm-up phase.
* **Performance Improvement**: Demonstrates significant stability improvements on GLUE benchmarks compared to the baseline AdaLoRA.

## File Structure

The repository focuses on the core implementation of the improved algorithm:

* `/adalora/*`: The modified AdaLoRA implementation with magnitude-aware logic.
* `run_glue.py`: The training script adapted with the necessary callbacks for dynamic rank updates.

---

### ⚠️ Important Note for Usage

**If you want to run this code, please use Hugging Face's `peft` library and replace the corresponding files with the code provided here. This repository only contains the modified AdaLoRA implementation and does not include the full library environment.**

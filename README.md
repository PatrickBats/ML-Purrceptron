# Multi-Layer Purrceptron

Deep Learning approach to cat breed classification comparing from-scratch CNN training versus transfer learning with ResNet50.

## Overview

This project investigates whether a custom CNN can learn fine-grained visual distinctions entirely from scratch, or if pretrained ImageNet features are essential for cat breed classification with limited data.

**Dataset:** 20,000 balanced images across 8 cat breeds (Bengal, Bombay, British Shorthair, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese)



### From-Scratch CNN


**Model:** 5-layer CNN (13M parameters)
**Result:** 79.81% validation accuracy

### Transfer Learning Model

**Model:** ResNet50 fine-tuned (23.5M parameters)
**Result:** 86.49% validation accuracy












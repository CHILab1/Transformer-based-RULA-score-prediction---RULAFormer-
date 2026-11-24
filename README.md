# Transformer-based-RULA-score-prediction -RULAFormer 

# RULAFormer

**Transformer-based RULA score prediction**

RULAFormer is a research project that uses a lightweight Vision Transformer (ViT) to estimate Rapid Upper Limb Assessment (RULA) scores from images of working postures.

## Overview

RULA is an ergonomic assessment method that considers the posture of the upper limbs, neck, trunk, and legs, as well as factors such as muscle use and external load.

RULAFormer formulates score prediction as a supervised computer vision task. The model learns from images of working postures paired with reference RULA scores.

| Component | Description |
| --- | --- |
| Input | An image of a person performing a work task |
| Model | A lightweight Vision Transformer |
| Output | A predicted RULA score |

## Research Objective

The project evaluates how accurately a lightweight Vision Transformer can estimate RULA scores from visual data for preliminary ergonomic screening.

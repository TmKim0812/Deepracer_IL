# Deepracer_IL

## Overview

This repository contains the policy learning pipeline for an autonomous driving project based on **Imitation Learning (Behavior Cloning)**.

The perception (vision) module is developed separately by another team.  
This repository focuses on building a **policy network** that maps perception features and vehicle state information to driving actions.

The design is modular and flexible, allowing different input representations and policy architectures.

---

## System Pipeline

The overall control pipeline is structured as follows:


Perception (Vision Process)
│
▼
Input Features (z_t)

Previous Action (a_{t-1})

Vehicle State (s_t: speed, direction, etc.)
│
▼
Policy Network
(MLP / GRU / RNN)
│
▼
Actions
(Steer / Throttle / Straight / etc.)


---

## Input

At each timestep **t**, the policy receives:

- `z_t` — Feature representation from the perception module  
- `a_{t-1}` — Previous action  
- `s_t` — Vehicle state information (e.g., speed, heading, direction)

The policy is designed to be **feature-agnostic**.  
As long as the perception module provides a consistent feature tensor, the policy can process it.

---

## Policy Architecture

The policy network maps inputs to control outputs.

Possible model choices include:

### 1. MLP (Simple Regression)
- Direct mapping from input features to actions  
- Suitable when temporal dependency is limited  

### 2. GRU / RNN (Temporal Modeling)
- Captures time-series dependencies  
- Improves stability in dynamic driving scenarios  
- Useful for smoother and more robust control

The final architecture can be selected based on task complexity and real-time constraints.

---

## Output

The policy produces driving commands such as:

- Steering angle  
- Throttle / Acceleration  
- Discrete actions (e.g., straight / turn / brake)

The exact action representation may vary depending on the deployment or simulation environment.

---

## Objective

The goal of this repository is to provide:

- A clean and modular policy pipeline  
- Seamless integration with external perception modules  
- Support for both simple regression and temporal models  
- A foundation for supervised imitation learning training  
- Extensibility for future improvements  

---

## Notes

- The perception module is maintained separately.
- This repository focuses only on policy modeling and action generation.
- The interface between perception and policy should remain consistent throughout development.

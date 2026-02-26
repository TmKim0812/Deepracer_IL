# Deepracer Imitation Learning Task Overview

## System Pipeline

The overall control pipeline is structured as follows:


        +---------------------+
        |     Perception      |
        |  (Vision Process)   |
        +----------+----------+
                   |
                   v
        +---------------------+
        |        Policy       |
        |  (MLP / GRU / RNN)  |
        +----------+----------+
                   |
                   v
        +---------------------+
        |        Actions      |
        | (Steer / Throttle / |
        |   Straight / etc.)  |
        +---------------------+


---

## Input - not specified yet. These are just an example

At each timestep **t**, the policy receives:

- `z_t` — Feature representation from the perception module  
- `a_{t-1}` — Previous action  
- `s_t` — Vehicle state information (e.g., speed, heading, direction)

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

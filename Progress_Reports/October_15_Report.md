# Team 24 Progress Report - October 15th, 2025

## Proposal Feedback
Since you're using an existing environment, substantial algorithmic contributions and systematic experimentation are expected. Plan to implement and compare 3+ algorithms. Precisely define your MDP: observation, actions, episode length, reward calc, etc. Make sure to clearly articulate what's novel about your study. Ablation/generalization studies may be pertinent. Include complete MDP and specify algos and an incremental development plan in your Oct 15 progress report.

## MDP Definition
- **Observations:** 
    - The number of cars and pedestrians stopped (waiting to cross) in each crossing lane. This is one number per lane (total of $c$ crossing lanes).
    - The current time since the start of the episode ($t$ as a number of seconds).
    - Total cars/pedestrians gone through each crossing lane so far. This is one number per lane (total of $c$ crossing lanes).
    - The current phase in the sequence of traffic lights changes. This is always the same clockwise sequence with $p$ phases.
- **Action:** 
    - Update the duration of $p$ phase in the sequence of traffic lights changes starting from the next step.
    - Actions will be taken every 10 seconds (action interval).
    - Actions will be taken immediately (change the current step in traffic light sequence).
- **Episode Length:**
    - An episode will have fixed duration: 1 hour in duration with step length of 1 second.
- **Reward Calculation:**
    - Negative reward for car/pedestrian queue length in each crossing lane.
    - Negative reward for car/pedestrian waiting time.
    - Positive reward for each car/pedestrian crossed.
    - The reward will be calculated each time the action is taken using the information from last the action interval.
- **Other Information:**
    - The simulation will be set up to have cars spawning continuously.

## Algorithms Used (Minimum 3)

### 1. Q-learning (Lewis)
- A model free reinforcement learning algorithm, learns action-value function through direct interaction with the environment
- Main advantage is ease of implementation and clear to interpret
- Q-learning can efficiently learn adaptive control policies by discretizing traffic static like pedestrian waits and signal phases
- It can be a baseline for comparison with more advanced algorithms such as PPO and SAC
- The simplicity of Q-learning is suitable for us to experiment and debug in our SUMO-RL environment

### 2. Proximal Policy Optimization - PPO (Victor)
- Policy based RL algorithm thats learn stochastic policy directly, policy updates during learning to ensure accuracy
- It is able to dynamically adjust traffic light timings:
  - Managing all lanes at once, left, right and straight lanes
  - Managing car lanes and pedestrian crossing lanes
  - Changing and updating greenlight time duration based on traffic flow and density
  - Other complex decisions such as when to extend or shorten the greenlight duration depending if the traffic is heavy or light in certain lanes and directions
- Can handle discrete and continuous actions, suited for a traffic light scenario
- It's able to handle complex actions spaces and dynamically adjusts in real-time

### 3. Max-Pressure Algorithm - MP (Fatih)
- Max-Pressure is a deterministic and non-RL algorithm that, at each decision moment, selects the signal phase that most relieves congestion right now. It does this by comparing how many want to move (upstream demand) against how much space is available where they're going (downstream capacity), and it treats pedestrian crossings as first-class movements with their own pressure.
- **Measure current demand:**
  - Count vehicles on each incoming lane (optionally weight by waiting time).
  - Estimate how crowded each destination (downstream) lane is.
  - Count pedestrians at each crosswalk and how long they've waited.
- **Compute pressure for each movement:**
  - Vehicles (lane i → lane j): pressure ≈ upstream queue − downstream queue.
    - Pressure is high if many cars are waiting and the receiving lane is clear; low/negative if downstream is jammed.
  - Pedestrians (crosswalk c): pressure ≈ # waiting + small bonus × wait time.
    - Bigger and overdue groups yield higher pressure.
- **Score each feasible phase:**
  - A phase is a conflict-free bundle of vehicle movements and crosswalks that can go together.
  - Sum the pressures of all movements served by that phase → total phase pressure (instantaneous relief).
- **Choose and hold:**
  - Among allowed phases, pick the one with the highest total pressure.
  - If it differs from the current phase (and constraints allow), switch; otherwise extend the current phase briefly.
  - Repeat every control interval (e.g., 5s).

### 4. Automatic Goal Generation Model - AGGM (Gator)
- AGGM combines reinforcement learning with ontology-based reasoning to handle unseen situations in traffic control (emergency vehicles, congestion).
- **Key Components:**
  - Dual Reward System:
    1. Problem-specific reward: Minimizes vehicle waiting time (standard RL)
    2. State similarity reward: Reverts to familiar states during unseen situations
  - Ontology-Based Detection: Assigns importance weights to vehicle types:
    1. Ambulance = 10 (highest priority)
    2. Fuel truck = 5
    3. Trailer truck = 2
    4. Default vehicle = 0
- **How it Works:**
  1. Detects significant changes via importance weights
  2. If predefined goal exists → switches to it
  3. If no predefined goal → generates new goal to restore familiar state
- **Performance:**
  - 65-85% reduction in ambulance waiting time vs. baselines
  - 56-81% for fuel trucks
  - Maintains normal performance for default vehicles


## Study Novelty
- Papers have historically just been dealing with vehicles.
- Adding pedestrians to the TSC is the novel aspect we plan to add in the SUMO-RL environment. Pedestrians would have their own crosswalk lights. Cars will have dedicated right- and left-turn lanes.
- We are also adding the novel aspect of dynamical adjustments in real-time for green and red light durations based on traffic flow instead of fixed durations. 


## Incremental Development Plan
- Getting the simulation `test.py` running as a proof that we know how to use the GitHub code.
- Modify `single-intersection.net.xml` and `single-intersection.rou.xml` which are used to define the custom environment's map (network and routing information, respectively).
- Make it run with the simplest algorithm (perhaps rules-based one?) with minimal reward implementation.
- To demonstrate that the agent can interact, there has to be actions such as changing the traffic lights' timings. Not always the same timing (depending on traffic flow and pedestrians).
- Record the demo video (10 minutes, each member talks).
- Summarize progress in the October 30th progress report.
- Study and implement the algorithms
- Train agent and compare algorithms for efficiency 
- Implement the final chosen algorithm to work in the TSC scenario
- Fix bugs and errors
- Finalize and polish project

## Member Contributions
- ### Julien Larivière-Chartier:
    - **Last 2 weeks:** I set up the Git repo and progress report template. I installed SUMO-RL in WSLg with docker. Using this, I was able to launch the GUI tools to design maps and scenarios, but I was not able to run the test simulation. I think this is due to the backend difference (TRACI vs Libsumo). I contributed to precisely defining our MDP.
    - **Next 2 weeks:** I will investigate why it did not run and fix it such that we can use SUMO-RL for our Env Demo for October 27th. 

- ### Gator Guo:
    - **Last 2 weeks:** Researched the Automatic Goal Generation Model (AGGM) and its application to handling unseen traffic situations. Studied ontology-based reasoning approaches and how they integrate with reinforcement learning for adaptive traffic control. Contributed to algorithm selection and understanding the dual reward system for emergency scenarios.
    - **Next 2 weeks:** Implement or assist with implementing the AGGM algorithm in SUMO-RL. Focus on integrating ontology concepts for vehicle priority weighting (emergency vehicles, fuel trucks). Test AGGM performance on unseen situations and assist with demo preparation and performance comparisons.

- ### Fatih Ozer:
    - **Last 2 weeks:** Focused on understanding model-based vs. model-free methods and how they apply to traffic signal optimization. Studied the Max-Pressure algorithm and adapted it conceptually to handle both vehicle and pedestrian flows. Assisted in defining the project’s MDP formulation (states, actions, rewards) and reviewed SUMO-RL structure to identify where to integrate new algorithms.
    - **Next 2 weeks:** Implement the Max-Pressure baseline and verify its performance within SUMO-RL for both vehicle and pedestrian cases. Develop visualizations (plots and metrics) to monitor queue lengths, waiting times, and overall traffic efficiency. Assist in preparing the experiment setup for comparing reinforcement learning algorithms against non-RL baselines.

- ### Victor Wang:
    - **Last 2 weeks:** Reading papers to decide on which algorithms to use, finding the possible primary algorithm that we might use (PPO algo). Contribution to the novelty of the project, adding pedestrians and crossing lane signals. 
    - **Next 2 weeks:** Focus on getting the environment ready for the demo, familiarizing myself with SUMO-RL, study the PPO algorithm and how it can be adjusted and implemented for our TSC scenario

- ### Lewis He:
    - **Last 2 weeks:** Studied the SUMO-RL framework and reviewed reinforcement learning algorithms suitable for traffic signal control. Selected and outlined the implementation plan for Q-learning baseline.
    - **Next 2 weeks:** Implement and test the Q-learning agent in the SUMO-RL environment. Assist with preparing the Environment Demo video and continue to find out other available algorithms that are suitable for us.

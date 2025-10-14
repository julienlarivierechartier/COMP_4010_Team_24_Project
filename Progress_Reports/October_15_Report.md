# Team 24 Progress Report - October 15th, 2025

## Proposal Feedback
Since you're using an existing environment, substantial algorithmic contributions and systematic experimentation are expected. Plan to implement and compare 3+ algorithms. Precisely define your MDP: observation, actions, episode length, reward calc, etc. Make sure to clearly articulate what's novel about your study. Ablation/generalization studies may be pertinent. Include complete MDP and specify algos and an incremental development plan in your Oct 15 progress report.

## MDP Definition
- **Observations:** 
    - The number of cars and pedestrians stopped (waiting to cross) in each crossing lane. This is one number per lane (total of $c$ crossing lanes).
    - The current time since the start of episode ($t$ as a number of seconds).
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
[Write here]

## Study Novelty Comment
[Write here]

## Incremental Development Plan
[Write here]

## Member Contributions
- ### Julien Larivière-Chartier:
    - **Last 2 weeks:** I setup the Git repo and progress report template. I installed SUMO-RL in WSLg with docker. Using this, I was able to launch the GUI tools to design maps and scenarios, but I was not able to run the test simulation. I think this is due to the backend difference (TRACI vs Libsumo). I contributed to precisely defining our MDP.
    - **Next 2 weeks:** I will investigate why it did not run and fix it such that we can use SUMO-RL for our Env Demo for October 27th. 

- ### Gator Guo:
    - **Last 2 weeks:** [Write here]
    - **Next 2 weeks:** [Write here]

- ### Fatih Ozer:
    - **Last 2 weeks:** [Write here]
    - **Next 2 weeks:** [Write here]

- ### Victor Wang:
    - **Last 2 weeks:** [Write here]
    - **Next 2 weeks:** [Write here]

- ### Lewis He:
    - **Last 2 weeks:** [Write here]
    - **Next 2 weeks:** [Write here]
# Team 24 Progress Report - October 30th, 2025

## Member Contributions

- ### Julien Larivière-Chartier:
    - **Last 2 weeks:**
    -  Read the SUMO-RL documentation to understand how to implement the custom Gym environment to include pedestrians. Wrote the first version of “custom_env.py” (commit 0f860de):
        - Subclass TrafficSignal to include pedestrian-related properties.
        - Subclass ObservationFuntion to include some pedestrian observations
        - Register a new reward function with CustomTrafficSignal.register_reward_fn() that was going to include pedestrians in rewards calculation.
        - Subclass SumoEnvironment to include initialization of CustomTrafficSignal 
    - Prepared the live demo (commit 5664dc3):  
        - Used Netedit GUI tools to draw the intersection (“.net.xml” file) that was used for the Env demo. Also created the first version of the “.rou.xml” file defining car and pedestrian flow.
        - Modified the CustomSumoEnvironment._start_simulation() function to add a stime.sleep(delay) to allow GUI initialization with pedestrians. This was needed for live demo because the original SumoEnvironment (which did not support pedestrians) crashed by accessing a GUI related variable that was not yet initialized because it took longer to initialize with pedestrians were defined in the “.rou.xml” file.
    - **Next 2 weeks:** Design a script to run the various algorithms and collect results for analysis. Ensure that the results generation requirements are respected with the rest of the team when helping them develop the algorithms (PPO, Max Pressure, QLearning).

- ### Gator Guo:
    - **Last 2 weeks:** Studied reinforcement learning fundamentals and reviewed the custom SUMO environment code structure. Helped test the environment demo and verified that pedestrian integration worked correctly with the traffic simulation.
    - **Next 2 weeks:** 
Collaborate with Lewis on Q-learning implementation. Assist with defining the state-action space, implementing the Q-table update mechanism, and testing algorithm performance.

- ### Fatih Ozer:
    - **Last 2 weeks:** Advanced the implementation of the Max-Pressure baseline, focusing on its core requirement of data collection. Investigated and implemented methods to observe and extract real-time simulation data (such as queue lengths and waiting times) from SUMO. This observational data is essential for calculating the "pressure" metric required by the algorithm.
    - **Next 2 weeks:** Complete the implementation of the Max-Pressure algorithm by integrating the data observation pipeline with the decision logic. Begin verifying the algorithm's performance in the SUMO-RL environment, debugging its behavior with both vehicle and pedestrian traffic.

- ### Victor Wang:
    - **Last 2 weeks:** Researched about the PPO algorithm and learned about the fundamental logic, determined and implemented the observation and reward functions, set up test environment for env demo
    - **Next 2 weeks:** Further research the PPO algorithm, try to implement it, initial training and debugging, and playing around with hyperparameters

- ### Lewis He:
    - **Last 2 weeks:** Focused on understanding and reviewing our environment setup and code structure, including how reset and step functions operate in the custom SUMO environment. Helped test the environment demo to ensure the agent interaction and reward functions worked as expected.
    - **Next 2 weeks:** Begin implementing the Q-learning algorithm. Integrate it with the current environment, test parameter configurations, and evaluate performance metrics like average waiting time and queue length.




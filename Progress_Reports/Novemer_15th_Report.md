# Team 24 Progress Report - November 15, 2025

## Member Contributions

- ### Julien Larivière-Chartier:
    - **Last 2 weeks:** I worked on an abstract class to streamline training and results generation for the different algorithms. I also worked on the training and evaluation loop that uses this new abstract class (commit  412c9597a74a7b761c3f7618264e30434b4a7c94)    
    - **Next 2 weeks:**  Finish the loop and add all algos in form my peers. Run the script

- ### Gator Guo:
    - **Last 2 weeks:** Implemented a state space analysis tool to support Q-learning development. The tool analyzes Q-table size requirements for different discretization strategies (5, 10, 15, 20 bins per feature), estimates memory usage, and provides recommendations for optimal bin selection. This directly supports the state-action space design needed for Q-learning implementation.
https://github.com/julienlarivierechartier/COMP_4010_Team_24_Project/commit/50b2bf5353889150983fb72f3fb0e43da0028a29


    - **Next 2 weeks:** Develop visualization tools to analyze and compare algorithm performance. Create plots for training progress (reward curves, waiting times, queue lengths) across Q-learning, Max-Pressure, and PPO. Build scripts to generate performance comparison charts and summary statistics for the final report.


- ### Fatih Ozer:
    - **Last 2 weeks:** Successfully implemented the Max-Pressure baseline algorithm, as planned. This involved writing the core logic to calculate "pressure" for each potential green phase and a selection function (select_max_pressure_action) to choose the optimal action. Here is the commit link for the implementation:
https://github.com/julienlarivierechartier/COMP_4010_Team_24_Project/commit/ff5b356f3c20017f9ddbf6e08698128c05f4b91a
    - **Next 2 weeks:** Complete the final verification and debugging of the Max-Pressure agent within the SUMO-RL environment. Run initial experiments to collect performance metrics. Focus on developing the planned visualizations (plots of queue lengths, wait times) to analyze its efficiency

- ### Victor Wang:
    - **Last 2 weeks:** Started implementation of core PPO components and structure, (actor-critic neural network, rollout buffer, and initial ppo update logic). Commits b728039, 
03f4be4, 4d7aff5.
    - **Next 2 weeks:** Finish the whole PPO implementation, writing training scripts, connect ppo agent with sumo environment, test hyperparaters, debugging, and compare with other algorithms.

- ### Lewis He:
    - **Last 2 weeks:** Implemented the Q-learning algorithm, integrated it with the current environment, tested multiple parameter configurations, and evaluated performance metrics such as average waiting time and queue length.
    - **Next 2 weeks:** Continue refining the Q-learning algorithm by optimizing the learning rate and exploration parameters. Conduct extensive testing to evaluate its performance under different traffic scenarios. Begin documenting the results and comparing them with other algorithms like Max-Pressure and PPO.






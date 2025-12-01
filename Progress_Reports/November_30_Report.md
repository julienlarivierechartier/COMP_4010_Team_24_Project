# Team 24 Progress Report - November 30th, 2025

## Member Contributions

- ### Julien Larivière-Chartier:
    - **Last 2 weeks:** 
        - I implemented fixed time and random baselines by wrapping them with `BaseAlgorithm`.(commits: 1f379f7aef2d44035c21c0ba3b6cd3436f0e50e6, 3d3afd1405c0ed4cfa870c10f5a300d78f313ca5, 91a7252eb116a86a7b15bc83dfd1097eb46e8e45)
        - I helped teammates to conform to `BaseAlgorithm` interface (commits: 
        41e41bd4ccb5bd8298db06883534ecc3d9b61537,
        dafa2b37e54a4ebf0e272aafb188d44c86ff88d8 )
        - I implemented a DQN agent with a decaying epsilon. (commits: 3d3afd1405c0ed4cfa870c10f5a300d78f313ca5, b7c8bd5a9274478eb9b615a62d29d0465696e241)
        - After a few trials, I noticed that baselines outperformed the RL agents and that it could be due to the same amount of traffic coming from each direction not providing diverse enough patterns for RL to learn something useful. Consequently, randomly assigning traffic phases is more effective than greedy action in the long run. To mitigate this, I wrote a script to generate route files programmatically to provide more diverse data for training and evaluation hoping that the RL agents would learn useful patterns and would beat randomness. I had to modify `run_experiments.py` and `BaseAlgorithm` to allow using these new route files because setting the route involves instantiating a new `gym.Env` every time (agents needed to be able to update their internal reference to `gym.Env`). (commits: 2efc06694d6013bcbee5ecf8a7ca25a04d908a25,
        d3461420e1381fbd8013ce3841d89f483499cfe3)
        - I wrote hyperparameter grids for the diverse algorithms and baselines and ran the code training and evaluating all algorithms and baselines with many hyperparameter combinations. I ran this overnight for results to be ready for the Results Demo (December 1st). (commits: ae3bff32fac268d426cd05a53be6fa269bc1f557, 6c6683f7262ef071a2ff33831cfdb52bfaf18088)
    - **Next week:** Finish documenting the code and work on a Readme to explain to the TA how to run/test the code. Work on the report: I would like to write about the `BaseAlgorithm` interface definition and how it allowed `run_experiments.py` to follow a structured approach to train and evaluate the different algorithms. I will write about the DQN RL algorithm and about fixed time and the random baselines. I will also write about the route files that were generated to allow the algorithms to learn from diversified episodes corresponding to specific traffic scenarios.

- ### Gator Guo:
    - **Last 2 weeks:** 
Created Q-learning model checker utility (check_qlearning_model.py) to verify and analyze trained Q-learning models by examining state space coverage, Q-value distributions, hyperparameter settings (learning rate, gamma, epsilon), and memory usage to validate training effectiveness and compare different configurations.
https://github.com/julienlarivierechartier/COMP_4010_Team_24_Project/blob/main/check_qlearning_model.py

    - **Next 2 weeks:** 
Analyze Q-learning training results and contribute to final project report. Prepare for final demo (December 1st).

- ### Fatih Ozer:
    - **Last 2 weeks:**Ran extensive tests on the Max-Pressure baseline to observe reward accumulation patterns, using these insights to debug and refine the algorithm within the “run_experiments.py” pipeline. Developed a comprehensive data visualization module (plot_results.py) that aggregates results from all agents to generate comparative performance graphs, specifically benchmarking RL algorithms against baseline algorithms (Fixed, Random, and Max-Pressure). Also, contributed to the team's Q-learning architecture by proposing a state discretization strategy to effectively map continuous observation spaces to finite state-action pairs. Commits: 0fb1ac530d52a19399010c9e19cefbf87d75704c, ef2853238d7c4e8a13062e9b56c445dbe9ff80d0
    - **Next 2 weeks:** Conduct the final set of comparative experiments to generate the official results for the project. Generate the final performance graphs for the report and presentation. Assist the team in compiling the final project documentation.

- ### Victor Wang:
    - **Last 2 weeks:** Finished the PPO implementation, tested and debugged, played around with the hyperparameters to get best training results, trained PPO with the help of Julien, checked for logic flaws. Commits: 61fbcf5c67818a393801363da84cb1e06e9356fd, a003776f501a13bf1c225fc0e30634f50f524952, ae038445fe6684bd87641a77b687885e48c92500, d1589c684f502d46cbec2628bf4ccbd60e84d77a. Modified custom_env to account for pedestrian crossing times so that min green time is enough for pedestrains to cross. Commits: 9e46cb4f3132f28e1cd46b1f17b8f24d5d3f2572
    - **Next 2 weeks:** Prepare to explain in the final demo, analyze training and evaluation results, compare rewards, go in depth in the final report.

- ### Lewis He:
    - **Last 2 weeks:** Completed Q-Learning algo implementation, test with different hyperparameters, tried to make sure the evaluation for Q-Learning is great and stable after at least 10 episodes.
Github repo commits: 8e62bb9b342f90e234700d44fb1ba5f8a94a2e49,
E2c6806b26310cc7be93f556f9299543f2ea4e30,
276e9710e60059b03b9246dc6620469860bb9440,
5b17744a4f67271c041ed3ba8c0a1c549d01dd78
    - **Next 2 weeks:** Writing Project Report, describe clearly why I choose Q-Learning through experiments and reasons. And writing different aspects of the results, to make the project report complete and reasonable.





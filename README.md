# COMP 4010 Team 24 Project

Traffic Signal Control (TSC) to optimize traffic flow for a single 4-way 
intersection with pedestrian crossings. Experiments conducted with different reinforcement learning algorithms with results comparison. 

## Setup (Components)
### SUMO
- For local installation, follow instructions [here](https://sumo.dlr.de/docs/Downloads.php).
- Note the installation directory and set environment variable `SUDO_HOME=/path/to/sumo`

### SUMO-RL
- Clone the GitHub and follow the installation instructions from [here](https://github.com/LucasAlegre/sumo-rl.git).
- Alternatively `pip install -r requirements.txt` should pull the above from git.

## Setup (Docker)
- A Ubuntu 22.04 container with X11 forwarding (GUI support) was put together to facilitate setup.
- For Windows (WSL) and Linux users with Docker and Docker Compose installed. This setup is configured to run on Nvidia runtime inside WSL for PyTorch to use GPU (need Nvidia drivers and GPU, CUDA >= 12.6 installed in WSL). Furthermore, the container is setup to use the C++ backend called `libsumo` and offers performance gains of up to 25x compared to the Windows alternative (TraCI Python API enabling the SUMO-GUI):
    - Launch `./sumo-rl_torch_docker/start.sh` and wait for the build to complete.
    - Attach to the container (`sumo-rl`) in VSCode or with the terminal. 
- **Notes:**
    - `SUDO_HOME=usr/share/sumo` : Where SUMO is installed (and all the extra tools)
    - `WORKSPACE=$(id -u)/project` : Location of the git repository for COMP 4010 project inside the container.
    - `/opt/sumo-rl` : Where SUMO-RL GitHub was cloned.
    - If attaching through VSCode, you may want to `export BROWSER=firefox` after attaching because it may have been overwritten automatically. Otherwise some SUMO tools requiring browser support may not work as expected (e.g. `python ${SUMO_HOME}/tools/osmWebWizard.py`).

## Run the Live Demo
This shows a GUI TSC simulation with random action selection.
```shell
python demo.py
```

## Run the Result Generation Script
This performs grid search for each algorithm defined in `ALGORITHMS` over all hyperparamters combinations defined in `PARAM_GRID`. The results will be stored under `Results/{timestamp}/` and will be organized under subdirectories with names corresponding to  algorithm and hyperparameter combinations. Example `Results\20251201_005709\ppo_lr_0.0001_gamma_0.99_clip_0.2_gae_lambda_0.95_K_8_entropy_coef_0.01`. Each subsfolder contains the training summary, evaluation summary as well as algorithm checkpoint.

```shell
python run_experiments.py
```
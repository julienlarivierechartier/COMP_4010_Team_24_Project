# COMP 4010 Team 24 Project

Traffic Signal Control (TSC) to optimize traffic flow for a single 4-way 
intersection with pedestrian crossings. Experiments conducted with different reinforcement learning algorithms with results comparison. 

## Setup (Components)
### SUMO
- For local installation, follow instructions [here](https://sumo.dlr.de/docs/Downloads.php).
- Note the installation directory and set environment variable `SUDO_HOME=/path/to/sumo`

### SUMO-RL
- Clone the GitHub and follow the installation instructions from [here](https://github.com/LucasAlegre/sumo-rl.git).

## Setup (Docker)
- A Ubuntu 22.04 container with X11 forwarding (GUI support) was put together to facilitate setup.
- For Windows (WSL) and Linux users with Docker and Docker Compose installed:
    - Launch `./sumo-rl_docker/start.sh` and wait for the build to complete.
    - Attach to the container (`sumo-rl`) in VSCode or with the terminal. 
- **Notes:**
    - `SUDO_HOME=usr/share/sumo` : Where SUMO is installed (and all the extra tools)
    - `WORKSPACE=$(id -u)/project` : Location of the git repository for COMP 4010 project inside the container.
    - If attaching through VSCode, you may want to `export BROWSER=firefox` after attaching because it may have been overwritten automatically. Otherwise some SUMO tools requiring browser support may not work as expected (e.g. `python ${SUMO_HOME}/tools/osmWebWizard.py`).
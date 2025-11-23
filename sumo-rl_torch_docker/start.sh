#!/usr/bin/env bash

# Script to setup all linux environemnt variables and write them to .env file before 
# building and launching the sumo-rl container with docker compose. Tested in WSL.

# Halt on any error
set -e

# Set current path to the directory in which the script is
cd "$(dirname "$0")"

# Detect UID/GID/USERNAME
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)

# ***IMPORTANT: Default LIBSUMO_AS_TRACI 
# Setting it to 0 means use TraCI (GUI allowed).
# Setting it to 1 means use LibSUMO (no GUI)
LIBSUMO_AS_TRACI=0

# Detect display for WSLg (GUI)
if [[ -z "$DISPLAY" ]]; then
  export DISPLAY=:0
fi

# Repo root = parent of docker/
PROJECT_ROOT=$(cd .. && pwd)

# Write the environment variables to .env file
cat > .env <<EOF
UID=${USER_ID}
GID=${GROUP_ID}
USERNAME=${USER_NAME}
DISPLAY=${DISPLAY}
PROJECT_ROOT=${PROJECT_ROOT}
LIBSUMO_AS_TRACI=${LIBSUMO_AS_TRACI}
WORKDIR=/home/${USER_NAME}/project
EOF

# Display debug info 
echo "Using UID=$USER_ID GID=$GROUP_ID USERNAME=$USER_NAME"
echo "Mounting $PROJECT_ROOT to /home/$USER_NAME/project"
echo "Starting docker compose..."

# Build the container with compose
docker compose up -d --build

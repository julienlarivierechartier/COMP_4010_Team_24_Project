FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 python3-pip python3-dev python3-venv \
    git wget curl sudo \
    x11-apps \  
    && rm -rf /var/lib/apt/lists/*

# Add SUMO PPA and install SUMO
RUN add-apt-repository ppa:sumo/stable -y && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# Set SUMO_HOME
ENV SUMO_HOME=/usr/share/sumo
ENV PATH="$SUMO_HOME/bin:${PATH}"

# Install SUMO-RL
RUN pip install --no-cache-dir sumo-rl gymnasium numpy pandas matplotlib stable-baselines3
RUN git clone https://github.com/LucasAlegre/sumo-rl /opt/sumo-rl && \
    pip install -e /opt/sumo-rl

# Build-time args for dynamic user
ARG USERNAME
ARG UID
ARG GID
ARG WORKDIR

# Create matching host UID/GID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to user
USER $USERNAME
WORKDIR $WORKDIR

CMD ["/bin/bash"]

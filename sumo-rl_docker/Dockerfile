# Dockerfile to install all dependencies to run SUMO and all tools + SUMO-RL in Linux 
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 python3-pip python3-dev python3-venv \
    git wget curl sudo \
    x11-apps \  
    && rm -rf /var/lib/apt/lists/*

# Use python3 as python
RUN ln -s /usr/bin/python3 /usr/bin/python

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

# Below is to install firefox to be used for OpenStreetMap browsing https://leimao.github.io/blog/Ubuntu-2404-Docker-Firefox-Installation/
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        fonts-wqy-microhei && \
    apt-get clean

# System locale UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install Firefox and its dependencies
# https://support.mozilla.org/en-US/kb/install-firefox-linux
# https://www.mozilla.org/en-US/firefox/117.0/system-requirements/
RUN install -d -m 0755 /etc/apt/keyrings && \
    wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null && \
    gpg -n -q --import --import-options import-show /etc/apt/keyrings/packages.mozilla.org.asc | awk '/pub/{getline; gsub(/^ +| +$/,""); if($0 == "35BAA0B33E9EB396F59CA838C0BA5CE6DC6315A3") print "\nThe key fingerprint matches ("$0").\n"; else print "\nVerification failed: the fingerprint ("$0") does not match the expected one.\n"}' && \
    echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | tee -a /etc/apt/sources.list.d/mozilla.list > /dev/null && \
    echo 'Package: *\nPin: origin packages.mozilla.org\nPin-Priority: 1000' | tee /etc/apt/preferences.d/mozilla && \
    apt-get update && apt-get install -y --no-install-recommends \
        libpci-dev \
        libcanberra-gtk3-module \
        libgles2-mesa-dev \
        dbus-x11 \
        firefox && \
    apt-get clean

# Set default browser. This env commmand is useless if attaching from vscode 
# because it gets overriden automatically. You need to manually re-export after 
# attaching before launching osmWebWizard.py with:
# export BROWSER=firefox
# python ${SUMO_HOME}/tools/osmWebWizard.py
ENV BROWSER=firefox

# Create matching host UID/GID
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to user
USER $USERNAME
WORKDIR $WORKDIR

CMD ["/bin/bash"]

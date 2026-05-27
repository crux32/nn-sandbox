# Option A: For GPU Support (Recommended if you have an NVIDIA card)
FROM tensorflow/tensorflow:latest-gpu

# Option B: For CPU Only (Uncomment below and comment out the GPU line if no NVIDIA card)
# FROM tensorflow/tensorflow:latest

# Install any additional Linux utilities you might need
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

FROM python:3.11-slim

# Install system utilities and rendering dependencies
# - procps: provides essential tools like 'ps' and 'top' to monitor container processes
# - ffmpeg: required by matplotlib animation writers to export MP4 videos and high-quality GIFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install core scientific and visualization Python stack
RUN pip install --no-cache-dir jupyter numpy pandas matplotlib scipy seaborn tqdm imageio imageio-ffmpeg

# Set up a non-root user matching the host server's UID to prevent volume permission conflicts
# IMPORTANT: REPLACE '1016' WITH YOUR ACTUAL USER ID ON THE SERVER (Run 'id -u' in the server terminal to find it)
ARG UID=1016
RUN useradd -m -u ${UID} vselenaya

# Establish the workspace directory
WORKDIR /home/vselenaya/app
RUN chown -R vselenaya:vselenaya /home/vselenaya

USER vselenaya

# Document that Jupyter Notebook inside the container listens on port 8888.
# Note: EXPOSE does not actually publish the port to the host network. 
# To map it out and access Jupyter from your browser, use the '-p 8888:8888' flag during 'docker run'.
EXPOSE 8888

# Launch Jupyter Notebook exposed on all network interfaces inside the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

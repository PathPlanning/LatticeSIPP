# Server-Side Rendering & Benchmarking Environment

This directory contains the infrastructure configuration for offloading heavy pathfinding simulations, parallel benchmarks, and frame-by-frame visual rendering (MP4/GIF generation) to a high-performance remote server.

## Rationale
Generating animations for hundreds of dynamic obstacles is computationally expensive because Matplotlib must draw and commit scenes sequentially, frame by frame. Running this locally blocks resources and takes hours. By containerizing the environment, we can seamlessly deploy it on a multi-core remote server to run benchmarks in the background and utilize parallel worker pools (`ProcessPoolExecutor`).

---

## 1. Docker Environment Management

### Building the Image
To avoid permission mismatches when mounting host directories into the container, pass your server user's exact UID during the build phase (find your UID on the host by running `id -u`):

```bash
docker build --build-arg UID=$(id -u) -t pathfinding-jupyter-server .
```

### Running the Container

Run the container in detached (background) mode (`-d`) so it persists after you disconnect from your SSH session. Mount your working directory to keep notebooks and results synchronized with the host:

```bash
# Run this command from the directory containing the Dockerfile.
# The "\$(pwd)/../" path automatically mounts the entire repository root into the container.
docker run -d -p 8888:8888 \
  -v "$(pwd)/../:/home/vselenaya/app" \
  --name algorithm-benchmark-container \
  pathfinding-jupyter-server
```

#### Alternative: Running on Enterprise Servers (NVIDIA DGX, etc.)

If you are deploying this container on a remote high-performance server (like an NVIDIA DGX Station) and your local browser hangs indefinitely when connecting via SSH tunnel, the server's internal firewall might be blocking Docker's network bridge (leaving packets stuck in a `SYN_SENT` state).

To bypass complex routing restrictions, run the container using the host's network stack directly:

```bash
# Use --network host to bind Jupyter straight to the server's network interfaces
docker run -d \
  --network host \
  -v "$(pwd)/../:/home/vselenaya/app" \
  --name algorithm-benchmark-container \
  pathfinding-jupyter-server
```
*Note: When using `--network host`, the `-p 8888:8888` flag is omitted because the container automatically takes over port 8888 on the host machine. If this port is already occupied by another user on the server, you will need to change the `--port=8888` argument inside the Dockerfile `CMD` instruction.*


### Monitoring & Interacting

Since the container runs in the background, use the following commands to check logs (for example, to retrieve the Jupyter security token) or drop into a shell:

```bash
# View logs and retrieve the Jupyter access token
docker logs algorithm-benchmark-container

# Enter the running container's bash environment for debugging
docker exec -it algorithm-benchmark-container bash

# Stop the container and terminate all active internal processes
# Useful for a clean reset if too many parallel instances were spawned. 
# The container status will switch to 'Exited' in 'docker ps -a'
docker stop algorithm-benchmark-container

# Wake the container back up without automatically restarting the internal entrypoint processes
docker start algorithm-benchmark-container
```

---

## 2. Secure Access via SSH Tunneling (Port Forwarding)

Production or institutional servers typically block public traffic on standard ports like `8888`. Instead of opening hazardous firewall ports or dealing with self-signed SSL certificates inside Docker, use an encrypted SSH tunnel to forward the container's port directly to your local machine.

Run this command **on your local machine**:

```bash
ssh -L 9123:localhost:8888 user@server_ip_address
```

### How it works:

* `-L 9123:localhost:8888` binds port `9123` on your local computer to port `8888` on the remote server (which Docker maps to the container).
* Leave this terminal window open. You can now access Jupyter safely in your local web browser at: **`http://127.0.0.1:9123`** (paste the token copied from `docker logs`).
* Now all code, cells, and computations executed inside the opened notebook run entirely on the remote server's hardware, utilizing its CPU, RAM, and storage resources while your local machine only acts as a display.

---

## 3. Persistent Background Execution (`nohup`)

For long-running sequential benchmarks (e.g., executing the multi-map orchestration pipeline from `/python-benchmark`), raw interactive SSH session can be risky — if the network drops, the execution kills instantly.

To run scripts completely decoupled from your session inside the container terminal, use:

```bash
PYTHONUNBUFFERED=1 nohup python3 runner.py > benchmark_log.txt 2>&1 &
```

### Parameters Breakdown:

* `PYTHONUNBUFFERED=1`: Forces Python to flush its stdout/stderr streams instantly to disk. Without this, logs might sit in the buffer and won't appear in your log file until the script completes or crashes.
* `nohup`: "No Hang Up". Protects the running process from being terminated when the parent SSH session or terminal closes.
* `> benchmark_log.txt`: Redirects standard output stream to a file.
* `2>&1`: Redirects standard error (2) into standard output (1), keeping all logs consolidated in one file.
* `&`: Shifts the entire operation to a background job, returning control of the prompt to you immediately.

To monitor progress in real-time:

```bash
tail -f benchmark_log.txt
```

---

## 4. SSH Key Authentication Management

To ensure seamless login and secure automated scripting without typing passwords repeatedly, use explicit SSH key pairs.

### Generating a Isolated Project Key

Generate a modern, highly secure Ed25519 key pair with a custom filename identifying the target laboratory/server:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_ResearchLab -C "vselenaya@ResearchLab"
```

*Note: `-C` appends an internal text comment to the public key file to help identify the owner/context easily inside the remote `authorized_keys` file.*

### Connecting with a Specific Key

If you maintain multiple keys, explicitly point to your newly generated private key file using the identity flag (`-i`):

```bash
ssh -i ~/.ssh/id_ed25519_ResearchLab user@server_ip_address
```
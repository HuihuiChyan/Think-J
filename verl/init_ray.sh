nvidia-smi

if [ "$RANK" = "0" ]; then
        echo "Starting Ray head node..."
        ray start --head --dashboard-host=0.0.0.0
python3 -c '
import ray
import sys
import time

@ray.remote
def check_nodes():
    return True

# 确保Ray已经初始化
if not ray.is_initialized():
    ray.init(address="auto")

expected_nodes = sys.argv[1]
max_wait_time = 900  # 最多等待300秒
start_time = time.time()

print(f"Waiting for {expected_nodes} nodes to be ready...")
while time.time() - start_time < max_wait_time:
    nodes = ray.nodes()
    alive_nodes = sum(1 for node in nodes if node["alive"])
    print(f"Current number of nodes: {alive_nodes}/{expected_nodes}")
    
    if alive_nodes >= expected_nodes:
        print("All nodes are ready!")
        break
    
    time.sleep(5)

if time.time() - start_time >= max_wait_time:
    print("Timeout waiting for nodes to be ready")
    exit(1)
' $1
    else
        max_attempts=60
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if ray start --address="$MASTER_ADDR":6379; then
                echo "Successfully connected to head node"
                break
            else
                echo "Attempt $attempt/$max_attempts: Connection failed, retrying in 5s..."
                sleep 5
                attempt=$((attempt + 1))
            fi
        done
        
        if [ $attempt -gt $max_attempts ]; then
            echo "Failed to connect after $max_attempts attempts"
            exit 1
        fi
        sleep 999999
    fi
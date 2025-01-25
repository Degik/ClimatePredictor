#!/bin/bash

# Head node IP and port.
HEAD_NODE_IP="131.114.3.229"
HEAD_NODE_PORT="6379"

# Stop Ray on the head node.
echo "Stopping Ray on head node..."
ray stop

# Wait for 5 seconds.
sleep 5

echo "Running command on head node..."
ray start --head --node-ip-address=localhost

# Loop through worker nodes from n12 to n14.
echo "Stopping and restarting Ray on worker nodes..."
for i in {12..14}; do
  WORKER_NODE="n${i}.maas"
  echo "Processing worker node: ${WORKER_NODE}"

  # Stop Ray on the worker node.
  echo "  Stopping Ray..."
  ssh ubuntu@"${WORKER_NODE}" "/home/ubuntu/.local/bin/ray stop"

  # Wait for 5 seconds.
  sleep 5

  # Start Ray on the worker node, connecting to the head node and specifying resources.
  echo "  Starting Ray..."
  ssh ubuntu@"${WORKER_NODE}" "/home/ubuntu/.local/bin/ray start --address=${HEAD_NODE_IP}:${HEAD_NODE_PORT} --resources '{\"n${i}\": 3}'"
done

echo "Ray start complete"
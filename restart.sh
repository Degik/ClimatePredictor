#!/bin/bash

# Head node IP and port.
HEAD_NODE_IP="131.114.3.229"
HEAD_NODE_PORT="6379"

# Stop Ray on the head node.
echo "Stopping Ray on head node..."
ray stop

# Wait for 5 seconds.
#sleep 5

echo "Running command on head node..."
ray start --head --node-ip-address=localhost --resources='{"head": 1}'

# Loop through worker nodes from n12 to n14.
echo "Stopping and restarting Ray on worker nodes..."
for i in {12..14}; do
  WORKER_NODE="n${i}.maas"
  echo "Processing worker node: ${WORKER_NODE}"

  # Stop Ray on the worker node.
  echo "  Stopping Ray..."
  ssh ubuntu@"${WORKER_NODE}" "/home/ubuntu/.local/bin/ray stop"

  # Wait for 5 seconds.
  #sleep 5

  # Start Ray on the worker node, connecting to the head node and specifying resources.
  echo "  Starting Ray..."
  ssh ubuntu@"${WORKER_NODE}" "/home/ubuntu/.local/bin/ray start --address=${HEAD_NODE_IP}:${HEAD_NODE_PORT} --resources '{\"n${i}\": 6}'"
done

echo "Ray start complete"


# OPTIONAL: Delete checkpoints folders on worker nodes.
# After the restart, in case delete checkpoints folders from workers

# Loop through worker nodes from n12 to n14.
echo "Deleting checkpoints folders on worker nodes..."
for i in {12..14}; do
  WORKER_NODE="n${i}.maas"
  echo "Processing worker node: ${WORKER_NODE}"

  # Delete the checkpoints folder on the worker node.
  echo "  Deleting checkpoints folder..."
  ssh ubuntu@"${WORKER_NODE}" "rm -rf /home/ubuntu/davide_b/checkpoints"
done

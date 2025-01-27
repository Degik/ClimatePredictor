#!/bin/bash

echo "Updating head node git repository..."
git pull

# Loop through worker nodes from n12 to n14.
echo "Updating worker nodes git repositories..."
for i in {12..14}; do
  WORKER_NODE="n${i}.maas"
  echo "Processing worker node: ${WORKER_NODE}"

  #  Update the git repository on the worker node.
  echo "Updating..."
  ssh ubuntu@"${WORKER_NODE}" "cd /home/ubuntu/davide_b/ClimatePredictor_RL_FL && git pull"
done

echo "Update complete"
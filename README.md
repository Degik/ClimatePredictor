# ClimatePredictor

A repository exploring **climate forecasting** using both **Reinforcement Learning (RL)** and **Federated Learning (FL)** approaches. This project aims to demonstrate how different machine learning techniques can be applied to predict climate-related metrics while maintaining data privacy and collaborating across multiple devices or institutions.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)  
   - [Discovery Head](#discovery-head)  
   - [Federated Weight Aggregation](#federated-weight-aggregation)  
   - [Worker Nodes](#worker-nodes)  
   - [Scalability & Performance](#scalability--performance) 
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

---

## Features

- **Unified RL & FL Training Pipeline**: The system jointly trains reinforcement learning agents and federated models to enhance climate predictions.
- **Collaborative Learning Framework**: Simulates data distribution across multiple clients while maintaining privacy using FL.
- **Adaptive Decision-Making**: RL agents optimize climate-related actions based on aggregated FL models.
- **Hourly Climate Features & Missing Data Analysis**:
  - The dataset consists of hourly climate features used as inputs for the network, with some missing values:
    - HourlyVisibility (4%) missing
    - HourlyStationPressure (4%) missing
    - HourlyRelativeHumidity (4%) missing
    - HourlyWindDirection (4%) missing
    - HourlyWindSpeed (4%) missing
    - HourlyAltimeterSetting (6%) missing
    - HourlyWetBulbTemperature (5%) missing
    - HourlyDewPointTemperature (4%) missing
  - The **HourlyDryBulbTemperature** feature is used as the **target** for the predictor.

---

## Overview

In many real-world scenarios, climate data is collected by various weather stations or institutions spread across different locations. Sharing or centralizing such data can be challenging due to privacy regulations, ownership concerns, or infrastructure constraints.

This project integrates **Reinforcement Learning (RL)** and **Federated Learning (FL)** into a unified system for climate forecasting, enabling models to:
- Learn optimal climate predictions through RL-based decision-making.
- Train collaboratively on decentralized datasets without sharing raw data via FL.
- Combine RL and FL to enhance predictive performance while preserving data privacy.

To manage large-scale distributed training and parallel computation efficiently, the project leverages the **Ray framework** (2.40v), which enables scalable RL and FL implementations by distributing workloads across multiple nodes.

By merging these techniques, the project highlights:
- The benefits of integrating RL and FL in climate prediction pipelines.
- How local training and global model aggregation improve forecasting accuracy.
- Performance trade-offs between different learning strategies in decentralized environments.

---


## System Architecture

The system architecture is designed for distributed and federated training, where each weather station (node) locally processes its own data and only communicates model weights to the central node.

### Discovery Head

The **discovery head** (or head node) supervises the entire system. Its key responsibilities include:
- **Initializing and monitoring** worker nodes (weather stations).
- **Managing the training iterations**, collecting updated weights from each worker and aggregating them into a global model.
- **Providing fault tolerance**: if any node fails, the discovery head redistributes tasks among the remaining nodes.

With **Ray**, the discovery head can start, stop, or reassign training processes without halting the entire system.

#### Federated Weight Aggregation

Within the discovery head, there is a **Federated Aggregator** component that:
- **Collects model parameters** (weights) from each worker node.
- **Combines these parameters** (by simple arithmetic mean or other federated averaging strategies) to update the global model.

In the current prototype, the aggregation applies a straightforward unweighted averaging. In future iterations, more complex weighting strategies could be added (e.g., based on the number of samples or the quality of each node’s data).

### Worker Nodes

Each **worker node** corresponds to a weather station and:
- Operates independently of the others.
- Maintains its own local subset of data.
- Runs a **Proximal Policy Optimization (PPO)**-based RL training on hourly weather data.
- Periodically sends updated weights back to the discovery head and receives the newly aggregated global weights.

Workers also save **training checkpoints** locally, enabling them to resume from the last saved state in case of unexpected failures, reducing redundant computations.

### Scalability & Performance

The system has been tested with increasing configurations (1, 3, 9 nodes) deployed on different machines or on the same machine, measuring mean and total training times.  
In summary:
- **Increasing node count** improves robustness and leverages more data in parallel.
- **Distributing nodes** across different machines (rather than multiple workers on one machine) typically reduces resource contention, improving mean training time.
- As more nodes join, **aggregation operations** on the discovery head can become a bottleneck, requiring more efficient aggregation strategies or more powerful hardware.

Below is an example comparison of training and aggregation times:

| **Configuration**             | **Mean Training Time (s)** | **Total Training Time (s)** | **Aggregation Time (s)** |
|-------------------------------|----------------------------:|----------------------------:|--------------------------:|
| 1 node                        | 215.26                     | 215.26                     | 0.050                    |
| 3 nodes (1 per machine)       | 213.80                     | 225.51                     | 0.066                    |
| 3 nodes (same machine)        | 247.97                     | 262.83                     | 0.068                    |
| 9 nodes (3 per machine)       | 232.31                     | 280.18                     | 0.125                    |

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Degik/ClimatePredictor
   cd ClimatePredictor
   ```

2. **Create a Virtual Environment (Optional but Recommended)**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Make sure you have Python 3.10.12+ (suggested) installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare Your Data**  
   - Place your climate dataset in the `data/` folder (or update the paths in the configuration).
   - Ensure the data is correctly formatted (CSV only).
   - The dataset used in this project is sourced from: [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/datasets)

2. **Run Federated Server**  
   ```bash
   python FederateServer.py
   ```

## Contributing

Contributions to enhance the project are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/my-new-feature
   ```
5. Open a Pull Request describing your changes.

---

## License

This project is licensed under the MIT License. Feel free to use it as a starting point for your own work. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For questions or suggestions, please open an issue on GitHub or reach out via:
- **GitHub**: [@Degik](https://github.com/Degik)
- **Email**: [bulottadavide@gmail.com](mailto:bulottadavide@gmail.com)

---

# ClimatePredictor

A repository exploring **climate forecasting** using both **Reinforcement Learning (RL)** and **Federated Learning (FL)** approaches. This project aims to demonstrate how different machine learning techniques can be applied to predict climate-related metrics while maintaining data privacy and collaborating across multiple devices or institutions.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

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

# Room Occupancy Estimation

## Overview
This project develops a predictive model to estimate room occupancy using environmental sensors. By leveraging sensors for CO2, temperature, sound, and light, the model provides real-time occupancy data, which can be used to drive intelligent building systems in both residential and commercial settings. This approach aims at enhancing comfort and optimizing energy efficiency.

## Data
The data for this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation). It includes measurements from various sensors in a 6m x 4.6m room over a period of four days, collected every 30 seconds.

### Features
- **Temperature:** Readings from four sensors (Celsius)
- **Light:** Lux readings from four sensors
- **Sound:** Voltage readings from sound sensors
- **CO2:** PPM from a CO2 sensor
- **PIR:** Motion detection from two sensors

## Models
We explored several machine learning models:
- **Naive Bayes:** Utilizes a simple probabilistic model based on Bayes' theorem with strong feature independence assumptions.
- **Support Vector Machine (SVM):** Finds the optimal hyperplane for classification, used with gradient descent for optimization.
- **Logistic Regression:** Well-suited for binary classification problems, used here with a one-vs-all approach for multi-class classification.

## Results
The models' performance varied, with Logistic Regression showing the best overall accuracy and robustness across different classes.

## Usage
Details on how to set up and run this project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Omii2899/Room_Occupancy_Prediction.git
   ```
2. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```



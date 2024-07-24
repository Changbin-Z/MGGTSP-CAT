# MGGTSP-CAT
This Project is based on this paper : MGGTSP-CAT: Integrating Temporal Convolution and LSTM for Multi-Scale Greenhouse Gas Time Series Prediction via Cross-Attention Mechanism
## Introduction
The study focused on Beijing as the research area and constructed a real greenhouse gas dataset for predicting methane concentrations 48 hours in advance. In order to enhance the transparency and reproducibility of the model, the project provides the main code and dataset for reproducing the MGGTSP-CAT model.

## Dataset
The dataset used in this experiment includes ten features, such as total column water (from the surface to the top of the atmosphere), water vapor content, surface concentration of methane, average molar fraction of total methane columns, and carbon cycle. The carbon cycle describes the CO2 flux of terrestrial vegetation. The time series data spans from 20:00 on November 9, 2013, to 05:00 on January 1, 2021, Beijing time, comprising 20,876 data records with each record spaced three hours apart. The dataset contains 20,876 records, divided into a training set and a test set with a ratio of 8:2. The training set covers 16,535 data points, with the remaining 4,134 points designated as the test set.

## Note
To reproduce the MGGTSP-CAT model, additional preprocessing of the dataset and construction of multi-scale time series are required. The procedures are as follows:

**1.Missing Value Handling:** Use linear interpolation to fill in missing values.

**2.Outlier Detection:** Identify outliers by calculating the Z-Score for each data point. Replace outliers with the mean of surrounding data within a defined window, with a Z-Score threshold set to 2.

**3.Smoothing:** Apply wavelet transform for smoothing. The wavelet basis function used in the project is db8.

**4.Multi-Scale Time Series Construction:** After data preprocessing, construct time series at three different window sizes: 192, 96, and 48, to form multi-scale time series.



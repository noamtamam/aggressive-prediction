﻿# Project Title: **Brain Activity Analysis for Mice Aggression**



## Overview
This project analyzes brain activity in mice during aggression trials, distinguishing brain activity patterns between winning and losing mice. It utilizes a set of machine learning and statistical methods to identify key features of neural activity associated with aggression outcomes.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#Usage)
3. [Code Overview](#Code-Overview)
4. [Methods](#Methods)
5. [Visualization](#Visualization)

---


## Installation

To set up the environment, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/noamtamam/aggressive-prediction
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```
---

## Usage
1. **Configure paths and parameters:**
Define paths and constants in the config.py file, such as data file paths or ML algorithms configoration. 

2. **Run the analysis:**
```bash
python main.py
```
This will load the data, preprocess it, and perform data exploration, statistical analysis, and visualizations.


## Code-Overview

* #### Data Loading and Preprocessing:

  * load_data(): Main data loading function, processing trial information for each mouse.
  * load_timing_table(): Reads and processes the trial timing table.
  * load_trial_data(): Loads specific trial data based on CSV inputs and formats it for analysis.
* #### Data Exploration:

  * plot_heatmap(): Generates heatmaps of brain activity for each mouse, organized by trial.
  * box_plot_area_activity(): Creates box plots comparing calcium levels in brain regions for winners vs. losers.
  * plot_diffreneces(): Calculates and plots the differences in average brain region activity between winners and losers.
  
* #### Dimensionality Reduction and Modeling:

    * run_PCA(): Performs PCA on brain activity data to reduce dimensionality and extract key components.
    * find_best_model(): Uses grid search with cross-validation to optimize SVM model parameters. 
    * run_svm(): Trains an SVM model on the data, using PCA-transformed features if specified. 
    * compute_model_significance(): Conducts permutation testing to validate model significance.
* #### Feature Importance:

    * plot_features_importance(): Plots coefficients from the SVM model to illustrate feature importance for prediction.
## Methods
1.  PCA Analysis: PCA is used to reduce dimensionality and identify key components of variance in the data.

2.  SVM Classification: A Support Vector Machine (SVM) model is used to classify winners and losers based on brain activity, with a grid search optimizing the C parameter.

3.  Permutation Testing: This statistical method assesses the significance of the SVM model's accuracy by comparing it against randomized label results.

## Visualization
Various plots are generated to explore and understand the data:

* Heatmaps of brain activity for winners and losers.
* Box plots illustrating the distribution of brain region activity for winners vs. losers.
* PCA plots showing the distribution of brain activity in reduced dimensions.
* SVM feature importance plots showing significant brain regions contributing to aggression prediction.
All visualization results are saved in the graphs directory.


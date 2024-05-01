# K-Means Clustering Simulator App

This is a Dash web application for simulating and visualizing the K-Means clustering algorithm.\
[Github Link](https://github.com/yashmaniya0/K-Means-Clustering-Simulator)

## Introduction

The K-Means algorithm is an unsupervised machine learning technique used for clustering similar data points into groups or clusters. It aims to partition a dataset into K clusters, where each data point belongs to the cluster with the nearest mean (centroid). This application provides an interactive environment to understand and visualize the K-Means algorithm.

## Features

- **Dataset Generation**: Choose different shapes of datasets such as blobs, moons, and circles with varying sizes and numbers of clusters.
- **Centroid Initialization**: Select different methods for initializing centroids, including random initialization and K-Means++ initialization.
- **Step-by-Step Visualization**: Observe the step-by-step execution of the K-Means algorithm, including the expectation step and maximization step.
- **Inertia Plot**: Track the change in inertia (within-cluster sum of squares) over iterations to assess the convergence of the algorithm.
- **Silhouette Coefficient**: Evaluate the quality of clustering using the silhouette coefficient, which measures the cohesion and separation of clusters.
- **Play and Pause Animation**: Control the animation of the algorithm with play and pause buttons for better visualization.
- **Customizable Parameters**: Adjust parameters such as the number of clusters, dataset size, and maximum iterations to explore different scenarios.

## How to Run

To run the K-Means Clustering Simulator App locally, follow these steps:

1. **Clone Repository**: Clone this repository to your local machine using Git:

    ```bash
    git clone https://github.com/yashmaniya0/K-Means-Clustering-Simulator
    ```

2. **Navigate to Directory**: Navigate to the directory containing the cloned repository:

    ```bash
    cd K-Means-Clustering-Simulator
    ```

3. **Install Dependencies**: Install the required Python dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the App**: Run the Python script to start the Dash web application:

    ```bash
    python app.py
    ```

5. **Access the App**: Open a web browser and go to [http://127.0.0.1:8050](http://127.0.0.1:8050) (or the url mentioned in your terminal) to access the K-Means Clustering Simulator App.

## Usage

1. **Select Dataset Shape**: Choose the shape of the dataset from the dropdown menu (e.g., blobs, moons, circles).
2. **Adjust Parameters**: Set the dataset size, number of clusters, centroid initialization method, and maximum iterations using the sliders and dropdown menus.
3. **Generate Dataset and Centroids**: Click on the "Generate Data" and "Generate Centroids" buttons to create the dataset and initialize centroids.
4. **Run Simulation**: Start the simulation by clicking on the "Play" button to observe the step-by-step execution of the K-Means algorithm.
5. **Visualize Results**: Watch the animation of the algorithm, explore the inertia plot, and analyze the silhouette coefficient to understand the clustering process.
6. **Interact with Animation**: Use the play, pause, next step, and previous step buttons to control the animation and navigate through iterations.

## Dependencies

- Dash: For building interactive web applications.
- Plotly: For creating interactive and customizable plots.
- NumPy: For numerical computations and array manipulations.
- Scikit-learn: For implementing the K-Means algorithm and evaluating clustering metrics.
- Matplotlib: For plotting graphs and visualizations.
- OpenCV: For image processing tasks (optional).

## Authors

- Yash Maniya: [GitHub Profile](https://github.com/yashmaniya0)
- Tatvam Shah: [GitHub Profile](https://github.com/TatvamShah)
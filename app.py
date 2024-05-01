# K-Means Clustering Simulator App

# IMPORTS

import json
import numpy as np
import dash
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from data_modification import create_dataset, init_centroids, Kmeans_EStep, Kmeans_MStep, make_kmeans_viz_data
import ui


# INITIALIZATION

app = dash.Dash(__name__)

# APPLICATION

## BUILD DASH APPLICATION

app.layout =  ui.final_layout


#               CALLBACKS                 #

#                           CALLBACK 1 : UPDATE DATASET AND STORE IN HIDDEN DIV
@app.callback(Output('dataset-value', 'children'), 
            [Input('shape-dropdown', 'value'), Input('sample-slider', 'value'), Input('cluster-slider', 'value'), Input('regenData-button', 'n_clicks')])
def update_dataset(sampleShape, sample_size, n_clusters, regenData_n_clicks):
    # CREATE DATASET
    X = create_dataset(shape=sampleShape, sampleSize=sample_size, n_clusters=n_clusters)
    X = json.dumps(X.tolist())
    return X

#                        CALLBACK 2 : UPDATE CENTROIDS AND STORE IN HIDDEN DIV
@app.callback(Output('kmeansCentroids-value', 'children'), 
            [Input('init-dropdown', 'value'), Input('centroid-slider', 'value'), Input('dataset-value', 'children'), Input('regenCentroids-button', 'n_clicks')])
def update_kmeans_centroids(initMethod, n_centroids, dataset, regenCentroids_n_clicks):
    X = np.array(json.loads(dataset))
    centroids = init_centroids(X, k=n_centroids, initMethod=initMethod)
    centroids = json.dumps(centroids.tolist())
    return centroids

#                         CALLBACK 3 : UPDATE K-MEANS FRAMES, INERTIA, SILHOUETTE
global_kmeans_frames_counter = -1
global_frames = []
global_inertia = []
global_silhouette = tuple()
@app.callback(Output('kmeans_frames-counter', 'children'), 
            [Input('dataset-value', 'children'), Input('kmeansCentroids-value', 'children'), Input('iter-slider', 'value')])
def update_kmeans_frames(dataset, kmeans_centroids, max_iter):
    global global_kmeans_frames_counter, global_frames, global_inertia, global_silhouette

    # UPDATE COUNTER
    global_kmeans_frames_counter = global_kmeans_frames_counter + 1
    
    # LOAD DATASET & CENTROIDS **
    X = np.array(json.loads(dataset))
    centroids = np.array(json.loads(kmeans_centroids))
    n_centroids = centroids.shape[0]
    labels = [-1]*X.shape[0]
    # RUN K-MEANS 
    inertia_hist = []
    kmeans_frames = []
    kmeans_frames.append(make_kmeans_viz_data(X, labels, centroids, ui.cluster_colors))
    for i in range(max_iter):
        # Expectation Step
        labels =   Kmeans_EStep(X, centroids)
        kmeans_frames.append(make_kmeans_viz_data(X, labels, centroids, ui.cluster_colors))
        # Maximization Step
        centroids = Kmeans_MStep(X, centroids, labels)
        kmeans_frames.append(make_kmeans_viz_data(X, labels, centroids, ui.cluster_colors))
        # Compute Inertia
        inertia = 0
        for j in range(n_centroids):
            inertia = inertia + np.power(np.linalg.norm(X[labels==j,:]-centroids[j], axis=1), 2).sum()
        inertia_hist.append(inertia)
    global_inertia = inertia_hist
    global_inertia = (global_inertia, KMeans(n_clusters=len(centroids)).fit(X).inertia_)
    # COMPUTE SILHOUETTE **
    silhouette_vals = silhouette_samples(X, labels)
    global_silhouette = ({k:silhouette_vals[labels==k] for k in range(n_centroids)}, silhouette_score(X, labels))
    # CREATE FRAMES *
    global_frames = [{'data':kmeans_frames[0], 'layout':{**layout, 'title':'Intialization...'}}]
    global_frames = global_frames +  [{'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Expectation')}} if idx%2==0  
                                    else {'data':d, 'layout':{**layout, 'title':'Step {} : {}'.format(idx//2+1, 'Maximization')}}
                                    for idx,d in enumerate(kmeans_frames[1:])]

    return global_kmeans_frames_counter

#                                   CALLBACK 4 : UPDATE K-MEANS GRAPH
global_curr_step = 0
global_prev_clicks = 0
global_next_clicks = 0
global_restart_clicks=0
global_num_intervals = 0
global_frames_ctr = 0

layout = dict(
    xaxis = dict(zeroline=False, showgrid=False, showline=True, showticklabels=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', range=[-17,17]),
    yaxis = dict(zeroline=False, showgrid=False, showline=True, showticklabels=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', range=[-17,17]),
    margin = {'t':10, 'b':30, 'l': 40}, 
    plot_bgcolor = "rgba(30, 41, 83, 1)"  
)

@app.callback(Output('kmeans-graph', 'figure'), 
            [
            Input('nextStep-button', 'n_clicks'), Input('prevStep-button', 'n_clicks'), Input('restart-button', 'n_clicks_timestamp'), 
            Input('kmeans_frames-counter', 'children'), Input('interval-component', 'n_intervals')
            ])
def update_kmeans_graph(nextStep_n_clicks, prevStep_n_clicks, restart_n_clicks, frames_counter, n_intervals):
    global global_prev_clicks, global_next_clicks, global_curr_step, global_restart_clicks, global_num_intervals, global_frames_ctr

    if not nextStep_n_clicks: 
        nextStep_n_clicks=0
    if not restart_n_clicks: 
        restart_n_clicks=0
    if not prevStep_n_clicks: 
        prevStep_n_clicks=0
    if not n_intervals: 
        n_intervals = 0


    if (global_restart_clicks != restart_n_clicks) or (global_frames_ctr != frames_counter):
        global_restart_clicks = restart_n_clicks
        global_frames_ctr = frames_counter
        global_curr_step = 0
        d = global_frames[global_curr_step]['data']
        fig = dict(data=d, layout=layout)
        return fig

    elif (global_next_clicks != nextStep_n_clicks) or (global_num_intervals != n_intervals):
        global_next_clicks = nextStep_n_clicks
        global_num_intervals = n_intervals
        global_curr_step = min(global_curr_step + 1, len(global_frames)-1)
        d = global_frames[global_curr_step]['data']
        fig = dict(data=d, layout=layout)
        return fig

    elif global_prev_clicks != prevStep_n_clicks:
        global_prev_clicks = prevStep_n_clicks
        global_curr_step = max(global_curr_step - 1, 0)
        d = global_frames[global_curr_step]['data']
        fig = dict(data=d, layout=layout)
        return fig     
    else:
        def kmeans(X, k, max_iters=100):
            # Randomly initialize centroids
            centroids = X[np.random.choice(X.shape[0], k, replace=False)]
            
            for _ in range(max_iters):
                # Assign each data point to the nearest centroid
                labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
                
                # Update centroids
                new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
                
                # Check for convergence
                if np.all(centroids == new_centroids):
                    break
                
                centroids = new_centroids
            
            return labels, centroids

        kmeans_iteration = lambda X, cent: (
        np.argmin(np.linalg.norm(X[:, np.newaxis] - cent, axis=2), axis=1),
        np.array([X[np.argwhere(1 == i)[:, 0]].mean(axis=0) for i in range(cent.shape[0])]))

    d = global_frames[global_curr_step]['data']
    fig = dict(data=d, layout=layout)
    return fig

#                           CALLBACK 5 : UPDATE STEP/ITERATION TEXT
@app.callback(Output('iter-text', 'children'), [Input('kmeans-graph', 'figure')])
def update_iter_text(kmeans_fig):
    text = global_frames[global_curr_step]['layout']['title']
    return text 

#                           CALLBACK 6 : DISABLE NB CLUSTERS FOR CERTAIN SHAPES
@app.callback(Output('cluster-slider', 'disabled'), [Input('shape-dropdown', 'value')])
def disable_component(shape):
    if shape in ['moon', 'circle', 'noStructure']:
        return True
    return False

#                       CALLBACK 7 : CONTROL ANIMATION USING PLAY/PAUSE BUTTONS
global_play_clicks = 0
global_pause_clicks = 0
@app.callback(Output('interval-component', 'interval'), [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')])
def play_kmeans(play_clicks, pause_clicks):
    global global_play_clicks, global_pause_clicks, currentStepIter
    if play_clicks==None: play_clicks=0
    if pause_clicks==None: pause_clicks=0
    if global_play_clicks != play_clicks:
        global_play_clicks = play_clicks
        return 1000
    return 3600*1000

#                                  CALLBACK 8 : STATE OF PLAY BUTTON
@app.callback(Output('play-button', 'style'), [Input('interval-component', 'interval')])
def update_play_state(interval):
    if interval == 1000:
        style = {'background-color':'#33C3F0', 'border-color':'white', 'marginLeft':45, 'marginTop':30, 'marginBottom':10, 'marginRight':5, 'width':60, 'font-size':10, 'color':'white', 'text-align': 'center', 'padding': 0}
    else:
        style = {'marginLeft':45, 'marginTop':30, 'marginBottom':10, 'marginRight':5, 'width':60, 'font-size':10, 'text-align': 'center', 'padding': 0}
    return style

#                                       CALLBACK 9 : INERTIA GRAPH
layout_inertia = dict(
    #title = 'K-Means Cost Function',
    xaxis = dict(title='Iteration', zeroline=False, showgrid=False, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    yaxis = dict(title='Inertia', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', gridcolor="silver", tickformat='.0f'),
    margin = {'t':10, 'l':60, 'r':10, 'b':40},
    showlegend = False,
    plot_bgcolor = "rgba(30, 41, 83, 1)"
)

@app.callback(Output('inertia-graph', 'figure'), [Input('kmeans_frames-counter', 'children')])
def update_inertia_graph(frames_counter):
    data1 = go.Scatter(
        x = list(range(1, len(global_inertia[0])+1)), 
        y = global_inertia[0],
        mode = 'markers+lines',
        marker = dict(color='white', size=10, line = dict(width=2, color='rgb(205,92,92,1)')),
        line = dict(color='rgba(205,92,92,1)'),
        name = 'Distorsion')
    
    data2 = go.Scatter(
        x = [len(global_frames)//2-1],
        y = [global_inertia[1]+0.1*(global_inertia[0][0]-global_inertia[1])],
        mode = 'text',
        text = 'Global Minimum',
        name = str(round(global_inertia[1])),
        hoverinfo = 'name',
        textfont = dict(
            color = "white"
        )
    )

    layout_inertia['shapes'] = [dict(type='line', x0=0, y0=global_inertia[1], x1=len(global_frames)//2+1, y1=global_inertia[1], line={'color':'white', 'width':1, 'dash':'dot'})]
    fig = dict(data=[data1, data2], layout=layout_inertia)
    return fig


class HarrisCornerDetector():
    def __init__(self, directory, window_size=3, threshold=10000):
        self.directory = directory
        self.window_size = window_size
        self.threshold = threshold
        self.images = self.load_images()

    def load_images(self):
        images = []
        for filename in sorted(os.listdir(self.directory)):
            img = cv2.imread(os.path.join(self.directory, filename))
            height, width = img.shape[:2]
            new_width = [600, 400][not images]
            new_height = int(height * (new_width / width))
            img = cv2.resize(img, (new_width, new_height))
            images.append(img)
        return images

    def sobel(self, image, axis):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
                          ) if axis == 'x' else np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        return convolve(image, kernel)

    def gaussian_blur(self, image, size):
        kernel = self.gaussian_kernel(size)
        return convolve(image, kernel)

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g / g.sum()

    def harris_corner_detection(self, image):
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        height, width = gray.shape

        Ix = self.sobel(gray, 'x')
        Iy = self.sobel(gray, 'y')

        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        Sxx = self.gaussian_blur(Ixx, self.window_size)
        Syy = self.gaussian_blur(Iyy, self.window_size)
        Sxy = self.gaussian_blur(Ixy, self.window_size)

        det = (Sxx * Syy) - (Sxy ** 2)
        trace = Sxx + Syy

        harris_response = det - 0.04 * (trace ** 2)

        corners = np.zeros_like(harris_response)
        corners[harris_response > self.threshold] = 255

        return corners

    def detect_corners(self):
        detected_corners = []
        for i, image in enumerate(self.images):
            corners = self.harris_corner_detection(image)
            detected_corners.append(corners)
        return detected_corners

    def detect_corners(self):
        detected_corners = []
        for i, image in enumerate(self.images):
            corners = self.harris_corner_detection(image)
            detected_corners.append(corners)
        return detected_corners

    def compare_with_opencv(self):
        opencv_detected_corners = []
        for i, image in enumerate(self.images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            window = max(3, self.window_size - 4)
            corners_opencv = cv2.cornerHarris(gray, window, 3, 0.1)
            corners_opencv = cv2.dilate(corners_opencv, None)
            opencv_detected_corners.append(corners_opencv)
        return opencv_detected_corners

    def plot_images(self, detected_corners, opencv_detected_corners):
        fig, axes = plt.subplots(
            nrows=len(self.images), ncols=2, figsize=(16, 50))

        for i, (image, corners, opencv_corners) in enumerate(zip(self.images, detected_corners, opencv_detected_corners)):
            image_with_corners = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_with_corners[corners != 0] = [255, 0, 0]
            axes[i, 0].imshow(image_with_corners, extent=[
                              0, 20*image.shape[1], 20*image.shape[0], 0])
            axes[i, 0].set_title(
                f'Custom Harris Corner Detection - Image {i+1}')

            image_with_opencv_corners = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_with_opencv_corners[opencv_corners >
                                      0.01 * opencv_corners.max()] = [255, 0, 0]
            axes[i, 1].imshow(image_with_opencv_corners, extent=[
                              0, 20*image.shape[1], 20*image.shape[0], 0])
            axes[i, 1].set_title(f'OpenCV Corner Detection - Image {i+1}')

            for ax in axes[i]:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

#                                   CALLBACK 10 : UPDATE SILHOUETTE GRAPH
layout_silhouette = dict(
    xaxis = dict(title='Silhouette Coefficient', zeroline=False, showgrid=True, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks', gridcolor="silver"),
    yaxis = dict(title='Cluster', zeroline=False, showgrid=False, showline=True, linecolor='#4B4D55', linewidth=2,  mirror='ticks'),
    margin = {'t':20, 'l':50, 'r':10, 'b':40},
    showlegend = False,
    plot_bgcolor = "rgba(30, 41, 83, 1)"
)

@app.callback(Output('silhouette-graph', 'figure'), [Input('kmeans_frames-counter', 'children')])
def update_silhouette_graph(frames_counter):
    data = []
    y_lower = 5
    nCentroids = len(global_silhouette[0])
    sample_size = len([x for sublist in global_silhouette[0].values() for x in sublist]) 
    for i in range(nCentroids):
        silhouette_vals = global_silhouette[0][i]
        silhouette_vals.sort()
        y_upper = y_lower + len(silhouette_vals)
        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=silhouette_vals,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=1, color=ui.cluster_colors[i]),
                                 fill='tozerox',
                                 fillcolor= ui.line_colors[i])
        data.append(filled_area)
        y_lower = y_upper + 5

    trace = go.Scatter(
        x = [global_silhouette[1]-.1],
        y = [.2*sample_size],
        mode = 'text',
        text = 'Avg. Score',
        textfont = dict(
            color="white"
        )
    )
    data.append(trace)
    layout_silhouette['yaxis']['range'] = [0, sample_size+5*(nCentroids+1)]
    layout_silhouette['shapes'] = [dict(type='line', x0=global_silhouette[1], y0=0, x1=global_silhouette[1], y1=sample_size+6*nCentroids, line={'color':'white', 'width':2, 'dash':'dot'})]

    fig = dict(data=data, layout=layout_silhouette)
    return fig


#               EXECUTION

if __name__ == '__main__':
    app.run_server(debug=False)
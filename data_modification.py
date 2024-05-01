# ********************************************************************************************************
#                                    HELPER FUNCTIONS DEFINITION
# ********************************************************************************************************

# ********************************************
#                   Imports
# ********************************************
import numpy as np
from sklearn import datasets
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, color
from sklearn.cluster import KMeans
from scipy.ndimage import generic_filter

# ********************************************
#          Dataset Creation Function
# ********************************************
def create_dataset(shape='gmm', sampleSize=200, n_clusters=3):
    """
    This function creates a dataset to apply K-means on.
    Inputs :
        - shape : shape of the dataset. Possible values : 'gmm', 'circle', 'moon', 'anisotropic', 'No Structure'. Default : 'gmm'.
        - sampleSize : number of data points. Default : 200.
        - n_clusters : number of clusters in the dataset. This is applicable only for shapes 'gmm' and 'anisotropic'.
    Output :
        - (sampleSize, 2) numpy array.
    """
    clusterStd = [0.5, 1, 1.3]*3
    clusterStd = clusterStd[:n_clusters]
    
    if shape=='gmm':
        X = datasets.make_blobs(n_samples=sampleSize, n_features=2, centers=n_clusters, cluster_std=clusterStd)[0]
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15
        
    elif shape=='circle':
        X = datasets.make_circles(n_samples=sampleSize, factor=.5, noise=.05)[0]
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15
        
    elif shape=='moon':
        X = datasets.make_moons(n_samples=sampleSize, noise=.1)[0]
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15

    elif shape=='anisotropic':
        transformations = {0:[[0.6, -0.6], [-0.4, 0.8]], 1:[[-0.7, -0.6], [0.6, 0.8]], 2:[[0.8, -0.1], [0.8, 0.1]]}
        X, y = datasets.make_blobs(n_samples=sampleSize, n_features=2, centers=n_clusters, cluster_std=clusterStd)
        for i in range(n_clusters):
            X[y==i] = np.dot(X[y==i], transformations[i%3])
        X = 5*X
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15
    else:
        X = 30*np.random.rand(sampleSize, 2)-15
        
    return X


# ********************************************
#       Centroid Initialization Function
# ********************************************
def init_centroids(X, k=3, initMethod='random'):
    """
    This function initializes the centroids for the K-means algorithm.
    Inputs :
        - X : numpy matrix representing the dataset used for clustering. This is needed for the K-means++ centroid initialization.
        - k : number of centroids (or prototypes) to create. Default : 3.
        - initMethod : initialization method. Possible values : 'random' and 'kmeans++'. Default : 'random'.
    Output :
        - (k, 2) numpy array.
    """
    if initMethod == 'random':
        # Random Initialization
        centroids = 30*np.random.rand(k, 2)-15
    else:
        # Kmeans++ Initialization
        indices = list(range(X.shape[0]))
        centroids = np.empty((k,2))
        centroids[0, :] = X[np.random.choice(indices), :]
        D = np.power(np.linalg.norm(X-centroids[0], axis=1), 2)
        P = D/D.sum()

        for i in range(k-1):
            centroidIdx = np.random.choice(indices, size=1, p=P)
            centroids[i+1, :] = X[centroidIdx, :]
            Dtemp = np.power(np.linalg.norm(X-centroids[-1, :], axis=1), 2)
            D = np.min(np.vstack((Dtemp, D)), axis=0)
            P = D/D.sum()
        
    return centroids

# ********************************************
#      K-means Expectation Step Function
# ********************************************
def Kmeans_EStep(X, centroids):
    """
    This function performs the Expectation Step of the K-means algorithm. It assigns each data point to its closest centroid.
    Inputs :
        - X : numpy array of the dataset such as the one returned by the create_dataset function.
        - centroids : numpy array of centroids such as the one returned by the init_centroids function.
    """
    k = centroids.shape[0]
    # Initialize Points-Centroid Distance Matrix
    centroidDistMat = np.empty((X.shape[0], k))

    # Compute Points-Centroid Distance Matrix
    for i in range(k):
        centroidDistMat[:, i] = np.linalg.norm(X-centroids[i,:], axis=1)
    
    # Infer Labels
    labels = centroidDistMat.argmin(axis=1) 
    return labels


class ImageManipulation:
    def __init__(self, image_path):
        # Constructor for the ImageManipulation class
        # It initializes the object with an image path or a numpy array of the image
        if not isinstance(image_path, str):
            self.image = image_path
            return
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.image_resized = None

    def resize_image(self, size=(64, 64)):
        # Method to resize the image
        self.image_resized = self.image.resize(size)

    def get_original_image(self):
        # Method to get the original image as a numpy array
        return np.array(self.image)

    def get_resized_image(self):
        # Method to get the resized image as a numpy array
        return np.array(self.image_resized)
    
    def convert_to_single_channel(self, image):
        # Method to convert an RGB image to a single-channel image
        height, width, _ = image.shape
        single_channel_image = np.zeros((height, width), dtype=np.uint32)

        for i in range(height):
            for j in range(width):
                # Convert each pixel's RGB values to a single integer value
                single_channel_image[i, j] = image[i, j, 0] * 1000000 + image[i, j, 1] * 1000 + image[i, j, 2]

        return single_channel_image

    def convert_to_rgb(self, single_channel_image):
        # Method to convert a single-channel image back to an RGB image
        height, width = single_channel_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Extract the individual components of the single-channel value
                r = single_channel_image[i, j] // 1000000
                g = (single_channel_image[i, j] % 1000000) // 1000
                b = single_channel_image[i, j] % 1000

                # Set the RGB values in the output image
                rgb_image[i, j] = [r, g, b]

        return rgb_image.astype(np.uint32)

    def max_frequency_filter(self, window_size):
        # Method to apply a max frequency filter to the image
        if self.image.shape[0] <= 128:
            return self.image
        image = self.convert_to_single_channel(self.image)

        def element_with_max_frequency(array):            
            # Function to find the element with the maximum frequency in a window
            unique_values, counts = np.unique(array, return_counts=True)
            max_frequency_index = np.argmax(counts)
            max_frequency_element = unique_values[max_frequency_index]
            return max_frequency_element

        # Apply the max frequency filter using generic_filter
        filtered_image = generic_filter(image, element_with_max_frequency, size=window_size)

        return self.convert_to_rgb(filtered_image)


class KMeansClustering:
    def __init__(self, num_clusters):
        # Constructor for the KMeansClustering class
        self.num_clusters = num_clusters

    def fit(self, image, feature_space):
        # Method to perform K-Means clustering on the image
        height, width, _ = image.shape
        pixel_values = np.reshape(image, (height * width, -1))

        if feature_space == "RGB":
            # Use RGB values as feature space
            pixel_values_with_positions = pixel_values
        elif feature_space == "RGBxy":
            # Add x and y positions as features along with RGB values
            x_positions, y_positions = np.meshgrid(np.arange(height), np.arange(width))
            scaler = 1
            x_positions *= scaler
            x_positions *= scaler
            pixel_values_with_positions = np.concatenate((pixel_values, x_positions.reshape(-1, 1), y_positions.reshape(-1, 1)), axis=1)
        else:
            raise ValueError("Invalid feature space. Use 'RGB' or 'RGBxy'.")

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(pixel_values_with_positions)
        labels = kmeans.labels_
        raw_img = kmeans.cluster_centers_[labels]
        raw_img = np.array([x[:3] for x in raw_img])
        segmented_image = raw_img.reshape((height, width, 3))  # Reshape back to original image dimensions
        
        # Apply max frequency filter
        res = ImageManipulation(segmented_image.astype(np.uint8))
        window = height // 50  # Define window size for max frequency filter
        return res.max_frequency_filter(window)


class RatioCutClustering:
    def __init__(self, num_clusters):
        # Constructor for the RatioCutClustering class
        self.num_clusters = num_clusters

    def _calculate_ratio_cut(self, adjacency_matrix, labels):
        # Method to calculate the ratio cut for a given clustering
        cut_ratio = 0
        for i in range(self.num_clusters):
            indices_i = np.where(labels == i)[0]
            indices_not_i = np.where(labels != i)[0]
            
            # Ensure that indices are within bounds
            indices_i = indices_i[indices_i < adjacency_matrix.shape[0]]
            indices_not_i = indices_not_i[indices_not_i < adjacency_matrix.shape[0]]
            
            # Calculate the cut and ratio, handling the case where the denominator might be zero
            if len(indices_i) > 0:
                cut = np.sum(adjacency_matrix[np.ix_(indices_i, indices_not_i)])
                ratio = cut / len(indices_i)
                cut_ratio += ratio
        
        return cut_ratio / self.num_clusters

    def _create_adjacency_matrix(self, image):
        # Method to create the adjacency matrix for the image
        # Convert the image to grayscale
        gray_image = color.rgb2gray(image)
        
        # Compute gradients in x and y directions
        gradient_x = np.gradient(gray_image, axis=0)
        gradient_y = np.gradient(gray_image, axis=1)
        
        # Compute edge weights as the magnitude of gradients
        edge_weights = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Convert edge weights to similarity values
        adjacency_matrix = np.exp(-edge_weights)
        return adjacency_matrix

    def fit(self, image):
        # Method to perform Ratio-Cut clustering on the image
        # Convert the image to numpy array if it's not already
        if not isinstance(image, np.ndarray):
            image = io.imread(image)
        
        # Get the shape of the image
        height, width, _ = image.shape
        
        # Reshape the image into a 2D array of pixels
        pixel_values = np.reshape(image, (height * width, -1))
        
        # Perform KMeans clustering to get initial clustering
        kmeans = KMeans(n_clusters=self.num_clusters)
        labels = kmeans.fit_predict(pixel_values)
        
        # Create adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(image)
        
        # Initialize best cut ratio and best labels
        best_cut_ratio = float('inf')
        best_labels = None
        
        # Perform optimization
        for _ in range(10):  # Number of iterations for optimization
            # Calculate cut ratio for current labels
            cut_ratio = self._calculate_ratio_cut(adjacency_matrix, labels)
            
            # Update best cut ratio and best labels if needed
            if cut_ratio < best_cut_ratio:
                best_cut_ratio = cut_ratio
                best_labels = labels.copy()
            
            # Randomly reassign labels for optimization
            random_labels = np.random.randint(0, self.num_clusters, size=len(labels))
            random_cut_ratio = self._calculate_ratio_cut(adjacency_matrix, random_labels)
            
            # Update labels if random labels have lower cut ratio
            if random_cut_ratio < cut_ratio:
                labels = random_labels
        
        # Check if best_labels is None
        if best_labels is None:
            # If best_labels is still None, use labels from KMeans clustering
            best_labels = labels.copy()
        
        res = np.reshape(best_labels, (height, width))
        
        # Generate segmented image by assigning original colors
        segmented_image = np.zeros_like(image)
        for i in range(self.num_clusters):
            segmented_image[res == i] = np.mean(image[res == i], axis=0)
        
        res = ImageManipulation(segmented_image)
        window = height // 50  # Define window size for max frequency filter
        return res.max_frequency_filter(window)


class ImagePlotting:
    def plot_images(self, original_image, resized_image, segmented_image_kmeans, segmented_image_ratio_cut,
                    num_clusters, feature_space):
        # Method to plot original and segmented images
        fig, axes = plt.subplots(1, 3, figsize=(24, 16))

        axes[0].imshow(resized_image)
        axes[0].set_title('Resized Original Image')
        axes[0].axis('off')

        fs1 = 'RGB'
        if feature_space:
            fs1 = feature_space
        axes[1].imshow(segmented_image_kmeans)
        axes[1].set_title(f'K-Means Clustering\nNumber of Clusters: {num_clusters} ({fs1} space)')
        axes[1].axis('off')

        if feature_space:
            axes[2].imshow(segmented_image_ratio_cut)
            axes[2].set_title(f'Ratio-Cut Clustering\nNumber of Clusters: {num_clusters}')
            axes[2].axis('off')
        else:
            axes[2].imshow(segmented_image_ratio_cut)
            axes[2].set_title(f'K-Means Clustering\nNumber of Clusters: {num_clusters} (RGBxy space)')
            axes[2].axis('off')
        plt.show()


class SegmentationComparison:
    def __init__(self, image_paths, num_clusters_list):
        # Constructor for the SegmentationComparison class
        self.image_paths = image_paths
        self.num_clusters_list = num_clusters_list

    def compare_clustering(self, feature_space, resize_size=(256, 256)):
        # Method to compare clustering algorithms
        for idx, image_path in enumerate(self.image_paths):
            image_manipulation = ImageManipulation(image_path)
            image_manipulation.resize_image(size=resize_size)

            original_image = image_manipulation.get_original_image()
            resized_image = image_manipulation.get_resized_image()

            image_plotting = ImagePlotting()

            for num_clusters in self.num_clusters_list:
                # Perform K-Means and Ratio-Cut clustering
                segmented_image_kmeans = KMeansClustering(num_clusters).fit(resized_image, feature_space)

                ratio_cut_clustering = RatioCutClustering(num_clusters)
                segmented_image_ratio_cut = ratio_cut_clustering.fit(resized_image)

                # Plot the images
                image_plotting.plot_images(original_image, resized_image, segmented_image_kmeans,
                                           segmented_image_ratio_cut, num_clusters, feature_space)



# ********************************************
#      K-means Maximization Step Function
# ********************************************
def Kmeans_MStep(X, centroids, labels):
    """
    This function performs the Maximization Step of K-means. It computes the new centroids given the current centroid assignment of data points.
    K-Means learning is done by iterating the Expectation and Maximization steps until convergence, or until max_iter is reached.
    Inputs :
        - X : numpy array of the dataset.
        - centroids : numpy array of the centroids.
        - labels : list or numpy array of the current cluster assignment of each data point.
    Output :
        - Numpy array of the same shape of centroids.
    """
    k = centroids.shape[0]
    Newcentroids = centroids
    # Compute values for New Centroids
    for i in range(k):
        if sum(labels==i)>0:
            Newcentroids[i, :] = X[labels==i, :].mean(axis=0)

    return Newcentroids


# ********************************************
#   Make K-means Visualization Data Function
# ********************************************
def make_kmeans_viz_data(X, labels, centroids, clusColorMap):
    """
    This function creates the Plotly traces of a given K-means setting (Data points, Centroids, labels of data points).
    Inputs :
        - X : numpy array of the dataset.
        - labels : list or 1D numpy array representing the current cluster assignment of data points.
        - centroids : 2D numpy array representing the centroids.
        - clusColorMap : a dictionary mapping a centroid number to its desired color.
    Output :
        - a list of traces to plot using Plotly.
    """

    clusLabelColors = list(map(clusColorMap.get, labels))
    
    # Num Clusters
    k = centroids.shape[0]

    data = []
    
    # IF WE'RE IN THE INITIALIZATION STEP (Coloring changes)
    if sum(labels)==-X.shape[0]:
        # Data Points
        tracePoints = go.Scatter(
                    x = X[:, 0],
                    y = X[:, 1],
                    mode = 'markers',
                    marker = dict(color='gray', size=10,
                                  line = dict(width=1, color='white')),
                    name = 'Data Points'
        )
        data.append(tracePoints)
        # Centroid-Point Lines
        centroidPoints = X.copy()
        for idx,i in enumerate(range(1, 3*centroidPoints.shape[0], 3)):
            centroidPoints = np.insert(centroidPoints, i, X.mean(axis=0), axis=0)
            centroidPoints = np.insert(centroidPoints, i+1, np.array([None, None]), axis=0)
        traceLines = go.Scatter(
                        x = centroidPoints[:, 0],
                        y = centroidPoints[:, 1],
                        mode = 'lines',
                        line = dict(color='white', width=.5),
                        name = 'Memberships'
        )
        data.append(traceLines)
        # Centroids
        traceCentroid = go.Scatter(
                        x = centroids[:, 0],
                        y = centroids[:, 1],
                        name = 'Centroids',
                        mode='markers',
                        marker = dict(color='gray', 
                                      size=20, symbol='circle', opacity=.8, line = dict(width=3, color='black'))
        )
        data.append(traceCentroid)
        return data
    
    # ELSE
    ## FIRST TRACE TYPE : DATA POINTS
    tracePoints = go.Scatter(
                    x = X[:, 0],
                    y = X[:, 1],
                    mode = 'markers',
                    marker = dict(color=[clusColorMap[i] for i in labels], size=10,
                                  line = dict(width=1, color='white')),
                    name = 'Data Points'
    )
    data.append(tracePoints)

    ## SECOND TRACE TYPE : LINES BETWEEN DATA POINTS AND CENTROIDS
        # Compute Array with Nones to link centroids to their respective points with lines
    centroidPoints = X.copy()
    for idx,i in enumerate(range(1, 3*centroidPoints.shape[0], 3)):
        centroidPoints = np.insert(centroidPoints, i, centroids[labels[idx], :], axis=0)
        centroidPoints = np.insert(centroidPoints, i+1, np.array([None, None]), axis=0)

        # Trace Lines
    traceLines = go.Scatter(
                    x = centroidPoints[:, 0],
                    y = centroidPoints[:, 1],
                    mode = 'lines',
                    line = dict(color='white', width=.5),
                    name = 'Cluster Lines'
    )
    data.append(traceLines)
    
    ## THIRD TRACE TYPE : CENTROIDS
    traceCentroid = go.Scatter(
                        x = centroids[:, 0],
                        y = centroids[:, 1],
                        name = 'Centroids',
                        mode='markers',
                        marker = dict(color=list(clusColorMap.values()), 
                                      size=20, symbol='circle', opacity=.8, line = dict(width=3, color='black'))
    )
    data.append(traceCentroid)

    return data
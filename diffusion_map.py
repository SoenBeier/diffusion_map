import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
import warnings
from math import*
import copy

from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import scipy.sparse.linalg as spsl

import time


def save_matrix(M,name = "test", folder_name = "diffusion_map"):
    time1 = time.time()
    
    df = pd.DataFrame(M,copy=True)
    df.to_csv(folder_name + "/" + name + ".csv")    
    
    time2 = time.time()
    print("saved matrix in ", time2-time1, " seconds.")
    

def create_helix(radius, height,num_wobels, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle values
    z = np.linspace(0, 0, num_points) + height*np.sin(theta*num_wobels)  # Height values
    x = radius * np.cos(theta)  # X-coordinates
    y = radius * np.sin(theta)  # Y-coordinates

    # Calculate the helix coordinates
    coordinates = np.column_stack((x, y, z))

    return coordinates, 'Black'

    
def standardize_matrix(matrix):
    matrix = copy.deepcopy(matrix)
    
    print("Mean and standard deviation before standardizing:")
    for n in range(matrix.shape[1]): #for all variables in dataset
        mean = np.mean(matrix[:,n])
        sd = np.std(matrix[:,n])
        print("Before standardizing: column: " + str(n) + ", mean = " + str(mean) + ", standard deviation = " + str(sd))
    
        for m in range(matrix.shape[0]): #adapt all entries
            matrix[m,n] = (matrix[m,n] - mean)/sd  
            
        mean = np.mean(matrix[:,n])
        sd = np.std(matrix[:,n])        
        print("After standardizing: column: " + str(n) + ", mean = " + str(mean) + ", standard deviation = " + str(sd))
        
    return matrix    
    
def calculate_distance_matrix(X, methode = "sklearn"):
    time1 = time.time()
    
    X = copy.deepcopy(X)
    
    if methode == "sklearn":
        D = pairwise_distances(X, metric='euclidean')
        
    time2 = time.time()
    print("calculated distances in ", time2-time1, " seconds.")
    
    return D

def guess_epsilon_Cameron(D):
    ''' see "Diffusion Map" lecture notes from Maria Cameron '''
    #D = copy.deepcopy(D)
    
    squared_D = np.square(D)
    
    #search row minima
    row_minima = []
    for row in tqdm(range(D.shape[0]), desc = "Compute..."):
        squared_D[row,row] = float('Inf') #set diagonal entries to Inf, so they will not be a minima
        row_minima.append(min(squared_D[row,:]))
    
    #initial guess is two times the mean of row minimas
    epsilon = 2*np.mean(row_minima)        
    return epsilon    
    
def calculate_kernel_matrix(D, epsilon, kernel = "gaussian"):
    time1 = time.time()
    
    squared_D = np.square(D)

    K = np.exp(-squared_D/epsilon)
            
    time2 = time.time()
    print("calculated kernel matrix in ", time2-time1, " seconds.")
    
    return K
    
    
def only_take_nearest_neighbours(K,number_neighbours = 10):
    time1 = time.time()
    #K = copy.deepcopy(K)
    
    #delete entries which are below a treshold
    for i in tqdm (range(K.shape[0]), desc= "Compute..." ):
        considered_row = copy.deepcopy(K[i,:])
        considered_row.sort()
        threshold = considered_row[-number_neighbours]

        K[i,:][K[i,:]<threshold] = 0

                
    #keep a symetric matrix              
    K = np.maximum(K, K.transpose())
    
    time2 = time.time()
    print("only keep nearest neighbours in ", time2-time1, "seconds.")
    
    return K           
                
def normalizing_rows(K):
    time1 = time.time()
    
    r = np.sum(K, axis=0) # Sum of every row of K
    Di = np.diag(1/r) # Degree matrix
    P = np.matmul(Di, K)

    time2 = time.time()
    print("normalization in ", time2-time1, "seconds.")
    
    
    print("Sums of first rows after normalization: ")
    for rows in range(20):
        print(sum(P[rows,:]), end=","),
        
    return P
    
    
def calculate_diffusion_map(P,num_eigenv=6):
    time1 = time.time()

    eigenvalues, eigenvectors = spsl.eigs(P, k=num_eigenv, which='LR')
    print("calculated eigenvalues & eigenvectors")
    
    #sort eigenvalues and eigenvectors,delete first eigenvector
    ix = eigenvalues.argsort()[::-1][1:]
    eigenvalues = np.real(eigenvalues[ix])
    eigenvectors = np.real(eigenvectors[:, ix])
    
    #compute diffusion map
    diffusion_map = eigenvalues * eigenvectors
    
    time2 = time.time()
    print("calculated diffusion map in ", time2-time1, " seconds.")
    
    return diffusion_map, eigenvalues, eigenvectors

    
def plot_eigenvalues(eigenvalues,num_eigenv=6):
    plt.plot(list(range(len(eigenvalues)))[0:num_eigenv:1],list(eigenvalues)[0:num_eigenv:1],'*')
    plt.xlabel('index eigenvalue')
    plt.ylabel('eigenvalue')
    plt.show()
    
    
def plot_diffusion_map(diffusion_map, dimensions = 2,color = 'Black',marker = "."):
    fig = plt.figure()
    
    if dimensions == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel('Diffusion Component 3')
        ax.scatter(diffusion_map[:, 0], diffusion_map[:, 1],diffusion_map[:, 2],marker = marker,c=color)

    if dimensions == 2:
        ax = fig.add_subplot()
        ax.scatter(diffusion_map[:, 0], diffusion_map[:, 1],marker = marker,c=color)
        
    ax.set_xlabel('Diffusion Component 1')
    ax.set_ylabel('Diffusion Component 2')
    ax.set_title('Diffusion Map Visualization')
    
    
def plot_matrix(matrix):
    None
    

def is_matrix_symetric(M):
    n_zeros = 0
    n_not_zeros = 0
    n_entries_not_same = 0
    differences = [0]
    
    for i in tqdm(range(M.shape[0]), desc = "Compute..."):
        for j in range(i):
            if M[i,j] != 0:
                n_not_zeros = n_not_zeros + 1
            
            if M[i,j] != M[j,i]:
                n_entries_not_same = n_entries_not_same + 1
                differences.append(abs(M[i,j]-M[j,i]))
    
    print("n_zeros: ", M.shape[0]*M.shape[1] - n_not_zeros)
    print("n_not_zeros: ", n_not_zeros)
    print("n_entries_not_same: ", n_entries_not_same)
    print("max(differences): ", max(differences))
    
    if n_entries_not_same == 0:
        return True
    else:
        return False
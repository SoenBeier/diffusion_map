#include the diffusion map class and all methods to calculate the diffusion map
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt

from scipy import sparse
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_distances


class diffusion_map:
    def __init__(self, data, data_df = None, epsilon = 1, N = 100, t = 1, standardize_data = False, normalize_data = False, num_eigenv = 300, guess_epsilon = True, eigen_solver = "sparse",metric = 'euclidean', D = None): #calculates diffusion map from the data(matrix) with a gaussian kernal with the width epsilon. Only N nearest neighbours will be considered. When N=='all' all neighbours will be considered which corresponds to the original definition. The data can be standardized or normalized before calculating the diffusion map. num_eigenv give the number of calculated eigenvalues and diffusion components. guess_epsilon give a first estimate for a correct epsilon (see Lafon2004). It can be usefull to use 'linalg' instead of a 'sparse' eigen_solver. In principal it is possible to use other metrices instead of the 'euclidean' to calculate the distances of the datapoints.

        
        self.data = data
        self.df = data_df
        self.epsilon = epsilon
        self.N = N
        self.t = t    

        
        if normalize_data and standardize_data:
            raise ValueError("Select Normalize or Standardize only.")

        
        if standardize_data:
            #Standardize data, so that the mean of each column is 0 and the standard deviation is 1 (mean 0, sd 1)
            self.data = standardize_matrix(self.data)

            
        if normalize_data:
            #normalize the data so that the values are between 0 and 1
            self.data = normalize_matrix(self.data)

            
        #Compute Euclidean distances between data points (create Distance Matrix D)
        if type(D) == type(None):
            D = calculate_distance_matrix(self.data, methode = "sklearn",metric = metric)
            
        self.D = D

        
        #calculate initual guess for epsilon
        if guess_epsilon:
            print("initial guess for $epsilon$", guess_epsilon_Cameron(D))
        
        
        #Calculate kernel matrix K
        K = calculate_kernel_matrix(D, epsilon, kernel = "gaussian")

        
        #Deleting minor entries of K
        if N != 'all':
            K = only_take_nearest_neighbours(K,number_neighbours = N)
            
            
        #set no zero entries to 1 if epsilon == Inf
        if epsilon == "Inf":
            for i in tqdm(range(np.shape(K)[0]), desc = "Compute kernel..."):
                for j in range(np.shape(K)[1]):
                    if K[i,j] != 0:
                        K[i,j] = 1  
                        
        self.K = K

        
        if num_eigenv != 0: #if one is only interested in K Matrix (for example when finding the right epsilon)one can set this value to 0
            self.Ms, trace, d12, d_12 = calculate_Ms(K)
            
            self.dmap, self.eigenvalues, self.phis, self.psis = calculate_diffusion_map(self.Ms, trace, d12, d_12, self.t, n_eigenvectors = num_eigenv, eigen_solver = eigen_solver)
            self.dmap = self.dmap[:,1:]
        

    def create_df_with_dmap(self):#create a dataframe with the original data saved in data_df and 9 of the new calculated diffusion components 
        num_columns = self.dmap.shape[1]
        
        common_elements = np.intersect1d(np.array(self.df.columns), np.array(["dc1","dc2","dc3","dc4","dc5","dc6","dc7","dc8","dc9"]))
        if common_elements.size > 0:   
            columns=["dc1_new","dc2_new","dc3_new","dc4_new","dc5_new","dc6_new","dc7_new","dc8_new","dc9_new"]
        else:
            columns=["dc1","dc2","dc3","dc4","dc5","dc6","dc7","dc8","dc9"]
            
        num_columns = min([num_columns,len(columns)])
        
        df_dmap = pd.DataFrame(self.dmap[:,0:num_columns],columns = columns[:num_columns])
        self.df = pd.concat([self.df, df_dmap], axis=1)


def standardize_matrix(matrix): #standardize the dataset (mean=0,standard deviation=1) 
    matrix = copy.deepcopy(matrix)
    
    print("Standardizing data")
    for n in range(matrix.shape[1]): #for all variables in dataset
        mean = np.mean(matrix[:,n])
        sd = np.std(matrix[:,n])
        
        if sd == 0:
            print("for column",n,"the sd is 0","row can not be standardized")
            continue
            
        for m in range(matrix.shape[0]): #adapt all entries
            matrix[m,n] = (matrix[m,n] - mean)/sd   
        
    return matrix  


def normalize_matrix(matrix): #normalize dataset to range [0,1]
    matrix = copy.deepcopy(matrix)
    
    print("Normalizing data")
    for n in range(matrix.shape[1]): #for all variables in dataset
        column_min = np.min(matrix[:,n])
        column_max = np.max(matrix[:,n])
    
        for m in range(matrix.shape[0]): #adapt all entries
            matrix[m,n] = (matrix[m,n] - column_min)/(column_max - column_min)   
        
    return matrix


def calculate_distance_matrix(X, methode = "sklearn",metric = 'euclidean',n_jobs=1): #calculate distances between all datapoints; D_ij is the distance between the datapoints i & j
    time1 = time.time()
    
    X = copy.deepcopy(X)

    if methode == "sklearn":
        D = pairwise_distances(X, metric=metric,n_jobs=n_jobs)
        
    time2 = time.time()
    print("calculated distances in ", time2-time1, " seconds.")
    
    return D


def guess_epsilon_Cameron(D):
    #see Stephane Lafon: Diffusion Maps and Geometric Harmonics, 2004
    
    squared_D = np.square(D)
    #squared_D = np.linalg.matrix_power(D,2)
    
    #search row minima
    row_minima = []
    for row in tqdm(range(D.shape[0]), desc = "Compute..."):
        squared_D[row,row] = float('Inf') #set diagonal entries to Inf, so they will not be a minima
        row_minima.append(min(squared_D[row,:]))
    
    #initial guess is two times the mean of row minimas
    epsilon = 2*np.mean(row_minima)        
    return epsilon   


def calculate_kernel_matrix(D, epsilon, kernel = "gaussian"): #use a kernal to get a stronger influence of the neighbourhood
    time1 = time.time()
    
    squared_D = np.square(D)

    K = np.exp(-squared_D/epsilon)
            
    time2 = time.time()
    print("calculated kernel matrix in ", time2-time1, " seconds.")
    
    return K


def only_take_nearest_neighbours(K,number_neighbours = 10): #only use the nearest neighbours of a datapoint and delete all other entries
    time1 = time.time()
    #K = copy.deepcopy(K)
    
    #delete entries which are below a treshold
    for i in range(K.shape[0]):
        considered_row = copy.deepcopy(K[i,:])
        considered_row.sort()
        threshold = considered_row[-number_neighbours]

        K[i,:][K[i,:]<threshold] = 0
       
    #keep a symetric matrix              
    K = np.maximum(K, K.transpose())
    
    time2 = time.time()
    print("only keep nearest neighbours in ", time2-time1, "seconds.")
    
    return K


def calculate_Ms(K):
    # Compute D squareroots and M_s
    r = np.sum(K, axis=0)
    trace = sum(r)
    d12 = np.diag(np.sqrt(r))
    d_12 = np.diag(1/np.sqrt(r))

    Ms = np.matmul(np.matmul(d_12, K),d_12)
    
    return Ms, trace, d12, d_12

    
def calculate_diffusion_map(Ms, trace, d12, d_12, t=1, n_eigenvectors = 300, eigen_solver = "sparse"): #calculate the diffusion map by calculating the eigenvalues and eigenvectors of the symmetrical matrix, see Nadler et al.: Diffusion maps, spectral clustering and eigenfunctions of Fokker-Planck operators, 2005
    
    n_eigenvectors = n_eigenvectors
    time1 = time.time()
    
    # Compute Spectrum of M_s, phi and psi
    if eigen_solver == "sparse":
        Ms_ = sparse.csr_matrix(Ms)    
        eigenvalues, eigenvectors = sparse.linalg.eigs(Ms_, k=n_eigenvectors, which='LR')
    elif eigen_solver == "linalg":
        eigenvalues, eigenvectors = linalg.eig(Ms,left = False,right = True)
    
    ix = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[ix])
    eigenvectors = np.real(eigenvectors[:, ix])

    phis = [(1/np.sqrt(trace))*np.dot(np.real(eigenvectors[:, i]),d12) for i in range(n_eigenvectors)]
    psis = [np.sqrt(trace)*np.dot(np.real(eigenvectors[:, i]),d_12) for i in range(n_eigenvectors)]

    if phis[0][0] < 0:
        phis[0] = -phis[0]

    # Compute diffusion map
    diffusion_map = (np.real(eigenvalues)**t)*np.array(psis).transpose()
    
    time2 = time.time()
    print("calculated diffusion map in ", time2-time1, " seconds.")
    
    return diffusion_map, eigenvalues, phis, psis







# from interface import *

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import os

def process(image_path, color_mode, process_mode, parameter):
    plt.rcParams["figure.figsize"] = [16, 8]
    A = imread(image_path)
    if process_mode == "original":
        if color_mode == "RGB":
            img = plt.imshow(A)
            plt.axis("off")
            plt.show()
            return
        else:
            X = np.mean(A, -1)
            img = plt.imshow(X)
            img.set_cmap(color_mode)
            plt.axis("off")
            plt.show()
            return
    elif process_mode in ["rank_ratio", "energy_ratio", "direct_rank_num"]:
        if color_mode == "RGB":
            A = np.transpose(A, (2, 0, 1))
            Xapprox = []
            for i in range(3):
                X = A[i]
                U, S, VT = svd(X, full_matrices = False)
                if process_mode == "direct_rank_num":
                    k = min(X.shape[0], X.shape[1])
                    assert parameter >= 1 and parameter <= k, "Parameter should be an integer between 1 and %d for current image in the mode 'direct_rank_num'!" % (k)
                    rank = parameter
                elif process_mode == "rank_ratio":
                    assert parameter > 0 and parameter <= 1, "Parameter should be a real number greater than 0 and no greater than 1 in the mode 'rank_ratio'!" 
                    k = min(X.shape[0], X.shape[1])
                    rank = int(parameter * k)
                elif process_mode == "energy_ratio":
                    assert parameter > 0 and parameter <= 1, "Parameter should be a real number greater than 0 and no greater than 1 in the mode 'energy_ratio'!" 
                    if parameter == 1:
                        rank = len(S)
                    else:
                        ratio_array = np.cumsum(S) / np.sum(S) 
                        rank = 1
                        while ratio_array[rank - 1] < parameter: rank += 1
                S = np.diag(S)
                Xapprox.append(U[:, :rank] @ S[:rank, :rank] @ VT[:rank, :])
            Xapprox = np.array(Xapprox, dtype = np.uint8)
            Xapprox = np.transpose(Xapprox, (1, 2, 0))
            img = plt.imshow(Xapprox)
            plt.axis("off")
            plt.show()
                
        else:
            X = np.mean(A, -1)
            U, S, VT = svd(X, full_matrices = False)
            if process_mode == "direct_rank_num":
                k = min(X.shape[0], X.shape[1])
                assert parameter >= 1 and parameter <= k, "Parameter should be an integer between 1 and %d for current image in the mode 'direct_rank_num'!" % (k)
                rank = parameter
            elif process_mode == "rank_ratio":
                assert parameter > 0 and parameter <= 1, "Parameter should be a real number greater than 0 and no greater than 1 in the mode 'rank_ratio'!" 
                k = min(X.shape[0], X.shape[1])
                rank = int(parameter * k)
            elif process_mode == "energy_ratio":
                assert parameter > 0 and parameter <= 1, "Parameter should be a real number greater than 0 and no greater than 1 in the mode 'energy_ratio'!" 
                if parameter == 1:
                    rank = len(S)
                else:
                    ratio_array = np.cumsum(S) / np.sum(S) 
                    rank = 1
                    while ratio_array[rank - 1] < parameter: rank += 1
            S = np.diag(S)
            Xapprox = U[:, :rank] @ S[:rank, :rank] @ VT[:rank, :]
            img = plt.imshow(Xapprox)
            img.set_cmap(color_mode)
            plt.axis("off")
            plt.show()
    else:
        return

   

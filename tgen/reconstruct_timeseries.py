#!/home/adriano/Escritorio/TFG/venv/bin/python3
import argparse

import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from datasets import load_dataset


# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)


import recurrence_plots
import activity_data

TEST_DIV = 1 #16 #--- TODO COMMENT OR SET = 1
SUBJECT_ID_TEST = 1600
WINDOW_ID_TEST = 0
NORMALIZE = 1
CLASS_LABEL_TEST = 0
TIMES_AUGM = 1



def varRP2(data, dim, TIME_STEPS=129):#dim:=x,y,z
    x = data
    
    
    s = []
    for i in range(len(x)-1):
    # for i in range(len(x)):

        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
    
        
    #print s
    # dimR = len(x)-1
    dimR = len(s)

    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            # if i==0 and j==0:
              # print("s[i]", s[i], "s[j]", s[j])
              # print(list(zip(s[i], s[j])))
              # print(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))))
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
            # R[i][j] = Distance2dim(s[i],s[j])
    return R
def construct_weighted_graph(rp):
    n = rp.shape[0]
    weighted_adjacency_matrix = np.zeros((n, n))
    
    nonzero_indices = np.nonzero(rp)
    for i, j in zip(*nonzero_indices):
        Gi = np.nonzero(rp[i])[0]
        Gj = np.nonzero(rp[j])[0]
        intersection = len(np.intersect1d(Gi, Gj))
        union = len(np.union1d(Gi, Gj))
        weighted_adjacency_matrix[i, j] = 1 - (intersection / union)
                
    return weighted_adjacency_matrix

# def calculate_shortest_path_matrix(weighted_adjacency_matrix):
#     G = nx.from_numpy_matrix(weighted_adjacency_matrix, create_using=nx.DiGraph)
#     shortest_path_matrix = nx.floyd_warshall_numpy(G)
#     return shortest_path_matrix +1e-10
def calculate_shortest_path_matrix(weighted_adjacency_matrix):
    shortest_path_matrix = dijkstra(weighted_adjacency_matrix, directed=True)
    return shortest_path_matrix + 1e-10

def reconstruct_time_series(shortest_path_matrix, ep=0.0, small_constant=1e-10):
    # Forzar la simetría en la matriz
    symmetric_shortest_path_matrix = 0.5 * (shortest_path_matrix + shortest_path_matrix.T)

    # Reemplazar los infinitos en la matriz con un valor grande pero finito
    finite_shortest_path_matrix = np.where(
        np.isfinite(symmetric_shortest_path_matrix),
        symmetric_shortest_path_matrix,
        ep  # Ajusta este valor según sea necesario
    )

    print("mfinita",finite_shortest_path_matrix)

     # Add a small constant to avoid division by zero
    finite_shortest_path_matrix += small_constant    

    # Apply MDS to get a 2D representation of the shortest path matrix
    mds = MDS(n_components=2, dissimilarity='precomputed')
    embedded_coords = mds.fit_transform(finite_shortest_path_matrix)

    #----
    # Calcular la matriz de covarianza
    cov_matrix = np.cov(embedded_coords, rowvar=False)

    # Calcular los valores y vectores propios de la matriz de covarianza
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Encontrar el índice del valor propio más grande
    max_eigenvalue_index = np.argmax(eigenvalues)

    # Seleccionar la columna correspondiente al valor propio más grande en la matriz embedded_coords
    selected_column = embedded_coords[:, max_eigenvalue_index][:, np.newaxis]

    return selected_column


def reconstruct_original_data(dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", TIME_STEPS=129):
    x_reconstructed = []
    sj_reconstructed = []
    y_reconstructed = []
    # classes = [np.argmax(y) for y in y_train]

    class_dir = f"{dataset_folder}numpies/train/windowed/"
    classes = [filename for filename in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir,filename))]
    for cl_i in  tqdm(range(len(np.unique(classes))), f"class"):
        cl = np.unique(classes)[cl_i]
        # subs = os.listdir(f"{dataset_folder}numpies/train/windowed/{cl}/")
        subs = [filename for filename in os.listdir(f"{dataset_folder}numpies/train/windowed/{cl}/") if os.path.isdir(os.path.join(f"{dataset_folder}numpies/train/windowed/{cl}/",filename))]

        for sbi in tqdm(range(len(subs)), f"subject"):
            sub = subs[sbi]
            #------------------------------------------------------------------
            # STEP 0: get original image
            #------------------------------------------------------------------
            all_windows = np.load(f"{dataset_folder}numpies/train/windowed/{cl}/{sub}/x.npy")
            # print("all_windows shape:", all_windows.shape)
            all_windows_reconstructed = []
            for window_id in tqdm(range(all_windows.shape[0]), f"[class-{cl}] window-of-sub-{sub}"):
                original_window = all_windows[window_id, :, :]
                df_original_plot = pd.DataFrame(original_window, columns=["x_axis", "y_axis", "z_axis"])
                df_original_plot["signal"] = np.repeat("Original", df_original_plot.shape[0])
                df_original_plot = df_original_plot.iloc[:df_original_plot.shape[0]//TEST_DIV-1,:]
                # print("df_original_plot",df_original_plot.head(128))
                # print("df_original_plot",df_original_plot.shape)
                # print(df_original_plot.describe())

                #------------------------------------------------------------------
                # STEP 1: generate Recurrence plot image
                #------------------------------------------------------------------
                img = recurrence_plots.SavevarRP_XYZ(original_window, sub, window_id, normalized = NORMALIZE, saveImage=False)
                img = np.array(img)
                # plt.imshow(img)
                # plt.show()

                # img = encode_signals_as_images(original_window[:,0], original_window[:,1], original_window[:,2])
                # plt.imshow(img)
                # plt.show()
                data_axis = []

                # print("img1", img1.shape, "img", img.shape)

                for axis in range(3):  # 0 >> x-accelerometer; 1 >> y-accelerometer; 2 >> z-accelerometer;
                
                    # #------------------------------------------------------------------
                    # # STEP 2: Reconstruction of time-series from RP- generate graph with edges
                    # #------------------------------------------------------------------
                    rp = img[:img.shape[0] // TEST_DIV, :img.shape[1] // TEST_DIV, axis]
                    # print("rp.shape", rp.shape)
                    weighted_adjacency_matrix = construct_weighted_graph(rp)
                    shortest_path_matrix = calculate_shortest_path_matrix(weighted_adjacency_matrix)
                    reconstructed_time_series = reconstruct_time_series(shortest_path_matrix, ep=0.0)
                    # print("reconstructed_time_series", reconstructed_time_series.shape)

                    # reconstructed_time_series = reconstructed_time_series[PC_idx,:,:]
                    data_axis.append(reconstructed_time_series)
                    # print("reconstructed_time_series", reconstructed_time_series.shape)

                #------------------------------------------------------------------
                # PLOT RECONSTRUCTION
                #------------------------------------------------------------------
                reconstructed = np.squeeze(data_axis).T#[PC_idx,:,:]
                # print("reconstructed:", reconstructed.shape)
                # print("reconstructed[:, 1]", reconstructed[:, 1])
                # reconstructed *= 5


                reconstructed_scaled = reconstructed.copy()
                # print("reconstructed scaled:", reconstructed_scaled.shape)

                #--- Normalizacion

                # for x in range(3):
                #   reconstructed_scaled[:, x] = np.interp(reconstructed[:, x], (reconstructed[:, x].min(), reconstructed[:, x].max()), (df_original_plot.values[:, x].min(), df_original_plot.values[:, x].max()))
                for x in range(3):
                    min_original, max_original = df_original_plot.values[:, x].min(), df_original_plot.values[:, x].max()
                    min_reconstructed, max_reconstructed = reconstructed[:, x].min(), reconstructed[:, x].max()
                    reconstructed_scaled[:, x] = (reconstructed[:, x] - min_reconstructed) * (max_original - min_original) / (max_reconstructed - min_reconstructed) + min_original
                    
                # print("reconstructed scaled:", reconstructed_scaled.shape)
                all_windows_reconstructed.append(reconstructed_scaled)
                x_reconstructed.append(reconstructed_scaled)
                y_reconstructed.append([cl])
                sj_reconstructed.append([sub])

                df_plot = pd.DataFrame(reconstructed_scaled, columns=["x_axis", "y_axis", "z_axis"])
                df_plot["signal"] = np.repeat("Reconstructed", df_plot.shape[0])
                # print("reconstructed:", df_plot.head(128))


                df_reconstruct = df_plot.copy()
                # print("df_original_plot.shape", df_original_plot.shape)
                # print("df_reconstruct.shape", df_reconstruct.shape)

                # df_reconstruct.iloc[:, :-1] = df_reconstruct.iloc[:, :-1] * TIMES_AUGM
                # print(df_reconstruct.shape
                df_all_plot = pd.concat([df_original_plot, df_reconstruct])
                # print("df_all_plot", df_all_plot.shape)
                # print(f"{dataset_folder}visualization/{cl}/time-series-reconstruction/{sub}/")
                # plot_reconstruct_time_series(df_reconstruct, cl, subject=sub, timestep_start_idx=window_id, saveFig=False, showPlot=True)

                recurrence_plots.plot_reconstruct_time_series(df_all_plot, cl, dataset_folder=dataset_folder,TIME_STEPS=TIME_STEPS, subject=sub, timestep_start_idx=window_id, saveFig=True, showPlot=False)
            #-- END --  for window_id in range(all_windows.shape[0]):
            os.makedirs(f"{dataset_folder}numpies/train/times-series-reconstruction/{cl}/{sub}/", exist_ok=True)
            np.save(f"{dataset_folder}numpies/train/times-series-reconstruction/{cl}/{sub}/x.npy" , np.array(all_windows_reconstructed))
    np.save(f"{dataset_folder}numpies/train/times-series-reconstruction/X_reconstructed.npy" , np.array(x_reconstructed))
    np.save(f"{dataset_folder}numpies/train/times-series-reconstruction/y_reconstructed.npy" , np.array(y_reconstructed))
    np.save(f"{dataset_folder}numpies/train/times-series-reconstruction/sj_reconstructed.npy" , np.array(sj_reconstructed))

def reconstruct_from_synthetic_images(dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/results/loso/fold_0/", visualization_savepath="/home/fmgarmor/proyectos/TGEN-timeseries-generation/results/loso/visualization_fold_0/", sampling="loso", TIME_STEPS=129):
    x_reconstructed = []
    y_reconstructed = []
    print(dataset_folder)
    print(os.listdir(dataset_folder))
    classes_filenames = [filename for filename in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, filename))]
    classes = [int(cl.split("_")[1]) if "_" else int(cl) in cl for cl in classes_filenames]
    
    print(classes_filenames)
    print(classes)
    
    for cl_i in  tqdm(range(len(np.unique(classes_filenames))), f"class"):
        cl_filename = np.unique(classes_filenames)[cl_i]
        cl = np.unique(classes)[cl_i]
        print("class:", cl)
        
        synth_dataset = load_dataset("imagefolder", data_dir=f"{dataset_folder}{cl_filename}")["train"]
        print("synth_dataset", synth_dataset.shape)
        all_windows_reconstructed = []
        for i, image in enumerate(synth_dataset):
            img = np.array(image["image"])
            data_axis = []
            for axis in range(img.shape[2]):  # 0 >> x-accelerometer; 1 >> y-accelerometer; 2 >> z-accelerometer;
            
                # #------------------------------------------------------------------
                # # STEP 2: Reconstruction of time-series from RP- generate graph with edges
                # #------------------------------------------------------------------
                rp = img[:img.shape[0] // TEST_DIV, :img.shape[1] // TEST_DIV, axis]
                # print("rp.shape", rp.shape)
                weighted_adjacency_matrix = construct_weighted_graph(rp)
                shortest_path_matrix = calculate_shortest_path_matrix(weighted_adjacency_matrix)
                reconstructed_time_series = reconstruct_time_series(shortest_path_matrix, ep=0.0)
                # print("reconstructed_time_series", reconstructed_time_series.shape)

                # reconstructed_time_series = reconstructed_time_series[PC_idx,:,:]
                data_axis.append(reconstructed_time_series)
                # print("reconstructed_time_series", reconstructed_time_series.shape)

            #------------------------------------------------------------------
            # PLOT RECONSTRUCTION
            #------------------------------------------------------------------
            reconstructed = np.squeeze(data_axis).T#[PC_idx,:,:]
            # print("reconstructed:", reconstructed.shape)
            # print("reconstructed[:, 1]", reconstructed[:, 1])
            # reconstructed *= 5


            reconstructed_scaled = reconstructed.copy()
            # print("reconstructed scaled:", reconstructed_scaled.shape)

            #--- Normalizacion

            # for x in range(3):
            #   reconstructed_scaled[:, x] = np.interp(reconstructed[:, x], (reconstructed[:, x].min(), reconstructed[:, x].max()), (df_original_plot.values[:, x].min(), df_original_plot.values[:, x].max()))
            # for x in range(img.shape[2]):
                # min_original, max_original = df_original_plot.values[:, x].min(), df_original_plot.values[:, x].max()
                # min_reconstructed, max_reconstructed = reconstructed[:, x].min(), reconstructed[:, x].max()
                # reconstructed_scaled[:, x] = (reconstructed[:, x] - min_reconstructed) * (max_original - min_original) / (max_reconstructed - min_reconstructed) + min_original
                
            # print("reconstructed scaled:", reconstructed_scaled.shape)
            # all_windows_reconstructed.append(reconstructed_scaled)
            x_reconstructed.append(reconstructed_scaled)
            y_reconstructed.append([cl])
            # sj_reconstructed.append([sub])

            df_plot = pd.DataFrame(reconstructed_scaled, columns=["x_axis", "y_axis", "z_axis"])
            df_plot["signal"] = np.repeat("Reconstructed", df_plot.shape[0])
            # print("reconstructed:", df_plot.head(128))


            df_reconstruct = df_plot.copy()
            # print("df_original_plot.shape", df_original_plot.shape)
            # print("df_reconstruct.shape", df_reconstruct.shape)

            # df_reconstruct.iloc[:, :-1] = df_reconstruct.iloc[:, :-1] * TIMES_AUGM
            # print(df_reconstruct.shape
            df_all_plot = df_reconstruct #pd.concat([df_original_plot, df_reconstruct])
            # print("df_all_plot", df_all_plot.shape)
            # print(f"{dataset_folder}visualization/{cl}/time-series-reconstruction/{sub}/")
            # plot_reconstruct_time_series(df_reconstruct, cl, subject=sub, timestep_start_idx=window_id, saveFig=False, showPlot=True)
            recurrence_plots.plot_reconstruct_time_series(df_all_plot, cl, dataset_folder=visualization_savepath,TIME_STEPS=TIME_STEPS, subject=i, timestep_start_idx=-1, saveFig=True, showPlot=False)
    os.makedirs(f"{dataset_folder}numpies/synthetic/{sampling}/times-series-reconstruction/", exist_ok=True)
    np.save(f"{dataset_folder}numpies/synthetic/{sampling}/times-series-reconstruction/X_reconstructed.npy" , np.array(x_reconstructed))
    np.save(f"{dataset_folder}numpies/synthetic/{sampling}/times-series-reconstruction/y_reconstructed.npy" , np.array(y_reconstructed))


def main():
    '''Examples of runs:
    - reconstruct only the original time-series
    $ nohup ./tgen/reconstruct_timeseries.py --reconstruct-original-data --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/ > reconstruction-original-data.log & 

    -reconstruct only from the synthetic images its time-series (LOSO strategy)
    $ nohup ./tgen/reconstruct_timeseries.py --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/results/ --sampling loso > reconstruction-synthetic-loso.log & 

    -reconstruct only from the synthetic images its time-series (LOTO strategy)
    $ nohup ./tgen/reconstruct_timeseries.py --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/results/ --sampling loto  > reconstruction-synthetic-loto.log & 

    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-folder', type=str, default="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", help='the data folder path')
    p.add_argument('--time-steps', type=int, default=129)
    p.add_argument('--n-folds', type=int, default=3)
    p.add_argument('--sampling', type=str, default="loso")
    p.add_argument('--reconstruct-original-data', action="store_true", help='reconstruct only original data; if not, from synthetic images reconstruct its time-series')


    args = p.parse_args()
    data_folder = args.data_folder
    time_steps = args.time_steps
    if args.reconstruct_original_data:
        reconstruct_original_data(data_folder, time_steps)
    else:
        for fold in range(args.n_folds):
            reconstruct_from_synthetic_images(f"{args.data_folder}{args.sampling}/fold_{fold}/", f"{args.data_folder}{args.sampling}/visualization_fold_{fold}/", args.sampling, args.time_steps)


if __name__ == '__main__':
    main()

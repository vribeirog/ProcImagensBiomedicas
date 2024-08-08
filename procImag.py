import os
import cv2
import pandas as pd
import numpy as np
import torch
import silhouette
from cuml.cluster import KMeans

# Função para carregar imagens
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Função para converter imagens de RGB para HSV
def convert_to_hsv(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]

# Função para aplicar blur nas imagens
def apply_blur(images, ksize):
    return [cv2.blur(img, (ksize, ksize)) for img in images]

#IMPLEMENTAÇÃO ORIGINAL DO KMEANS
# # Função para calcular a distância euclideana entre os dados e os centróides
# def distance(data, centroids, k):
#     cols = [f'C{i}' for i in range(1, k + 1)]
#     for i, col in enumerate(cols):
#         data[col] = np.sqrt((centroids[i][0] - data.R) ** 2 + (centroids[i][1] - data.G) ** 2 + (centroids[i][2] - data.B) ** 2)
#     data['Class'] = data[cols].idxmin(axis=1)
#     return data

# # Implementação do k-Means
# def kmeans(data, K):
#     print(10 * '-', f'k={K}\tDistance=euclidean', '-' * 10)
#     L = []
#     new_centroids = data.sample(K).values

#     data = distance(data.copy(), new_centroids, K)
#     old_centroids = new_centroids.copy()
#     new_centroids = np.array([data[data.Class == Class][['R', 'G', 'B']].mean().values for Class in data.loc[:, 'C1':f'C{K}'].columns])
#     i = 1
#     dist = np.linalg.norm(new_centroids - old_centroids)
#     print(f'Iteration: {i}\tDistance: {dist}')
#     while dist > 0.001:
#         L.append(dist)
#         data = distance(data, new_centroids, K)
#         old_centroids = new_centroids.copy()
#         new_centroids = np.array([data[data.Class == Class][['R', 'G', 'B']].mean().values for Class in data.loc[:, 'C1':f'C{K}'].columns])
#         dist = np.linalg.norm(new_centroids - old_centroids)
#         i += 1
#         print(f'Iteration: {i}\tDistance: {dist}')
#     print(f"k-Means has ended with {i} iterations")
#     return data, L

#IMPLEMENTAÇÃO COM KMEANS PARA GPU
# Função para calcular o silhouette_score médio
def calculate_silhouette_scores(images, k):
    chunk_size = 10000
    scores = []
    for img in images:
        
        imagem = img.reshape(-1, 3) #converte para um vetor com os pixeis
        
        df = pd.DataFrame({
        'R': img[:, :, 0].flatten(),
        'G': img[:, :, 1].flatten(),
        'B': img[:, :, 2].flatten()
        }, dtype=np.float32) #pandas dataFrame
        # print("df", df)        
        kmeansResult = KMeans(n_clusters=k, tol=0.001)
        kmeansResult.fit(df) #realiza o kmeans com a imagem
                        
        labels = kmeansResult.labels_ #pandas series
        labels_np = labels.to_numpy() #converte para numpy array
        # print("imagem", imagem)
        # print("labels", labels_np)
        
        X_tensor = torch.tensor(imagem, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_np, dtype=torch.int8)
        
        with torch.no_grad():
            if k > 1:
                score = silhouette.silhouette.score(X_tensor, labels_tensor)
                print("score = ", score)
                scores.append(score)
            else:
                scores.append(float('nan'))  
        
        torch.cuda.empty_cache()
    
        # exit()
    print()
    return np.nanmean(scores)  

# Função para calcular e exibir silhouette scores para diferentes k e tipos de imagem
def evaluate_images(images, label):
    # Calcular silhouette scores para k=2 e k=3 em RGB
    silhouette_scores_rgb = {}
    for k in [2, 3]:
        silhouette_scores_rgb[k] = calculate_silhouette_scores(images, k)
        print(f'Média do silhouette_score para k={k} (RGB) - {label}: {silhouette_scores_rgb[k]}')

    # Verificar o maior silhouette_score para RGB
    best_k_rgb = max(silhouette_scores_rgb, key=silhouette_scores_rgb.get)
    print(f'Melhor silhouette_score para RGB - {label}: k={best_k_rgb}, Score={silhouette_scores_rgb[best_k_rgb]}')

    # Converter imagens para HSV e calcular silhouette scores para k=2 e k=3 em HSV
    images_hsv = convert_to_hsv(images)
    silhouette_scores_hsv = {}
    for k in [2, 3]:
        silhouette_scores_hsv[k] = calculate_silhouette_scores(images_hsv, k)
        print(f'Média do silhouette_score para k={k} (HSV) - {label}: {silhouette_scores_hsv[k]}')

    # Verificar o maior silhouette_score para HSV
    best_k_hsv = max(silhouette_scores_hsv, key=silhouette_scores_hsv.get)
    print(f'Melhor silhouette_score para HSV - {label}: k={best_k_hsv}, Score={silhouette_scores_hsv[best_k_hsv]}')

    # Escolher o melhor modelo de cor e k
    best_color_model = 'RGB' if silhouette_scores_rgb[best_k_rgb] > silhouette_scores_hsv[best_k_hsv] else 'HSV'
    best_k = best_k_rgb if best_color_model == 'RGB' else best_k_hsv

    print(f'Melhor modelo de cor - {label}: {best_color_model}')
    print(f'Melhor k - {label}: {best_k}')

    # Aplicar blur e recalcular silhouette scores
    blurred_scores = {}
    for ksize in [11, 13]:
        blurred_images = apply_blur(images if best_color_model == 'RGB' else images_hsv, ksize)
        blurred_scores[ksize] = calculate_silhouette_scores(blurred_images, best_k)
        print(f'Média do silhouette_score para borramento {ksize} com k={best_k} ({best_color_model}) - {label}: {blurred_scores[ksize]}')

    # Imprimir todos resultados importantes
    print(f"\nResultados para imagens {label}:")
    print(f'Maior silhouette_score para RGB: k={best_k_rgb}, Score: {silhouette_scores_rgb[best_k_rgb]}')
    print(f'Maior silhouette_score para HSV: k={best_k_hsv}, Score: {silhouette_scores_hsv[best_k_hsv]}')
    print(f'Melhor modelo de cor: {best_color_model}')
    print(f'Melhor k: {best_k}')
    print(f'Maior silhouette_score para borramento 11 com k={best_k} ({best_color_model}): {blurred_scores[11]}')
    print(f'Maior silhouette_score para borramento 13 com k={best_k} ({best_color_model}): {blurred_scores[13]}')
    print("\n")

# Carregar as imagens da pasta ALL_IDB2
folder = r'ALL_IDB2'

# Verificar se a pasta existe
if not os.path.exists(folder):
    print(f'A pasta {folder} não existe.')
else:
    print(f'A pasta {folder} foi encontrada.')

images, filenames = load_images_from_folder(folder)
print(f'Imagens foram lidas.')

# Separar imagens com células não blastica (Y = 0) e células blastica (Y = 1)
no_blast_images = [img for img, filename in zip(images, filenames) if filename.split('_')[1][0] == '0']
blast_images = [img for img, filename in zip(images, filenames) if filename.split('_')[1][0] == '1']

# Avaliar imagens no_blast e blast
evaluate_images(no_blast_images, "no_blast")
evaluate_images(blast_images, "blast")
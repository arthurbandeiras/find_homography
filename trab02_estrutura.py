# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Arthur Bandeira Salvador

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):
    # Calculate centroid
    cent = np.mean(points, axis=0)
    # Calculate the average distance of the points having the centroid as origin
    distancias = np.zeros(len(points))
    
    for idx in range(len(points)):
      x_dis = points[idx,0]-cent[0]
      y_dis = points[idx,1]-cent[1]
      distancias[idx] = np.sqrt(x_dis**2 + y_dis**2)
    
    cent_avr_dist = np.mean(distancias)
    # Define the scale to have the average distance as sqrt(2)
    factor = np.sqrt(2) / cent_avr_dist
    # Define the normalization matrix (similar transformation)
    T = np.array([[factor, 0, -factor*cent[0]],
                  [0, factor, -factor*cent[1]],
                  [0,      0,               1]])
    # Normalize points
    homog_coord_points_t = np.column_stack((points, np.ones(len(points)))).T
    norm_pts = np.dot(T, homog_coord_points_t).T[:, :2]
    return T, norm_pts

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def DLT(pts1, pts2):
    # Add homogeneous coordinates
    pts1 = np.vstack((pts1.T, np.ones(pts1.shape[0])))
    pts2 = np.vstack((pts2.T, np.ones(pts2.shape[0])))
    # Compute matrix A
    for i in range(pts1.shape[1]):
        Ai = np.array([[0,0,0,*-pts2[2,i]*pts1[:,i],*pts2[1,i]*pts1[:,i]],
                    [*pts2[2,i]*pts1[:,i],0,0,0,*-pts2[0,i]*pts1[:,i]]])

        if i == 0:
            A = Ai
        else:
            A = np.vstack((A, Ai))

    # Perform SVD(A) = U.S.Vt to estimate the homography
    U,S,Vt = np.linalg.svd(A)
    # Reshape last column of V as the homography matrix
    H_matrix = Vt[-1,:] #in this case, reshape the last line of Vt

    return H_matrix.reshape((3,3))

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):

    # Normaliza pontos
    T1, norm1 = normalize_points(pts1)
    T2, norm2 = normalize_points(pts2)
    # Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
    # Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada
    """O que está comentado é feito na função DLT""" 
    H_normalizada = DLT(norm1, norm2)
    # Denormaliza H_normalizada e obtém H
    H = np.dot(np.linalg.inv(T2), np.dot(H_normalizada, T1))

    return H


########################################################################################################################
# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem 
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado 
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens
########################################################################################################################

"""M,_,_ = RANSAC(src_pts, dst_pts, 10, 72, 4)"""
def RANSAC(pts1, pts2, dis_threshold, N, Ninl):
    
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc 
    n_iterations = math.inf
    pts1_in = None
    pts2_in = None
    e = 0.5
    p = 0.99
    max_inliners = 0
    # Processo Iterativo
    while n_iterations > N:
        # Enquanto não atende a critério de parada
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        idx_sorted = np.random.choice(len(pts1), Ninl, replace=False)
        sorted_pts1 = pts1[idx_sorted]
        sorted_pts2 = pts2[idx_sorted]
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        estimated_H = compute_normalized_dlt(sorted_pts1, sorted_pts2)
        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado
        co_hom_pts1 = np.column_stack((pts1, np.ones(len(pts1))))

        #x' = H.x^t
        transformed_pts2 = np.dot(estimated_H, co_hom_pts1.T).T
        #obter a coluna 2 de transformed_pts2
        temp_col = transformed_pts2[:,2]
        #expandir a dimensão para dividir pelo vetor resultante
        temp_col = temp_col.reshape(-1,1)
        #assim, normalizamos as coordenadas homogêneas abaixo, dividindo
        #cada linha de transformed_pts2 pelo terceiro elemento da mesma linha
        transformed_pts2 /= temp_col
        #após a normalizaçao, a terceira coluna não se faz mais necessária,
        #então, podemos retirá-la
        """pega todas as linha, e as colunas 0 e 1"""
        transformed_pts2 = transformed_pts2[:, :2]

        dist = np.linalg.norm(pts2-transformed_pts2, axis=1)

        inliers = []
        for d in dist:
            if d < dis_threshold:
                inliers.append(True)
            else:
                inliers.append(False)
        inliers = np.array(inliers)

        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        # Atualiza também o número N de iterações necessárias

    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.
    H = compute_normalized_dlt(pts1_in, pts2_in)

    return H, pts1_in, pts2_in

def calculate_e(inliers, pts1):
    e = 1-((len(inliers))/(len(pts1)))
    return e

def calculate_N(p, e):
    val = ((np.log(1-p))//(np.log(1-((1-e)**4))))+1
    return val
########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT

MIN_MATCH_COUNT = 10
img1 = cv.imread('comicsStarWars01.jpg', 0)   # queryImage
img2 = cv.imread('comicsStarWars02.jpg', 0)        # trainImage

# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])#.reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])#.reshape(-1, 1, 2)
    
    #################################################
    M = compute_normalized_dlt(src_pts, dst_pts)
    #M = # AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################

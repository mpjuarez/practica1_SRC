#!/usr/bin/env python
# coding: utf-8

# # Sistema de Recomendación con KNN basado en el usuario
#  Librerías y carga de datos
#  URL ml-latest-small: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# In[24]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Cargar el dataset de calificaciones y películas
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')


# # Matriz Usuario-Película:

# In[25]:


# Crear la matriz usuario-película y llenar NaN con ceros
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convertir la matriz a dispersa para mejorar la eficiencia
user_movie_sparse_matrix = csr_matrix(user_movie_matrix)


# # Algoritmo KNN para encontrar usuarios similares
# Configuramos el algoritmo para calcular las distancias basadas en la similitud del coseno y seleccionar los n_neighbors usuarios más cercanos.

# In[26]:


# Configurar el modelo KNN para similitudes de coseno
#'algorithm',puede tomar cualquiera de estos valores ['auto', 'ball_tree', 'kd_tree', 'brute'
#algoritmo 'brure', de fuerza bruta, calcula la distancia al cuadrado de cada vector de características.

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
knn.fit(user_movie_sparse_matrix)


# # Predicción y Recomendación
# Definimos una función que, para un usuario específico, encuentra usuarios similares y calcula una calificación para cada película no vista por el usuario, basada en las calificaciones de sus vecinos más cercanos.

# In[27]:


# Función de recomendación de películas
def recomendar_peliculas(user_id, num_recommendations=5):
    user_index = user_id - 1  # Ajustar para índice de la matriz
    
    # Obtener vecinos cercanos al usuario
    distances, indices = knn.kneighbors(user_movie_sparse_matrix[user_index], n_neighbors=10)
    
    # Obtener calificaciones del usuario y películas no vistas
    user_ratings = user_movie_matrix.iloc[user_index]
    peliculas_no_vistas = user_ratings[user_ratings == 0].index

    # Si el usuario ha visto todas las películas, retornar sin recomendaciones
    if peliculas_no_vistas.empty:
        print(f"El usuario {user_id} ya ha visto todas las películas.")
        return pd.DataFrame(columns=['title'])
    
    # Almacenar los puntajes ponderados de recomendaciones
    recomendacion_scores = {}
    for pelicula in peliculas_no_vistas:
        score, suma_similitud = 0, 0
        for i in range(1, len(distances.flatten())):
            vecino_index = indices.flatten()[i]
            vecino_similitud = 1 - distances.flatten()[i]
            vecino_calificacion = user_movie_matrix.iloc[vecino_index][pelicula]
            if vecino_calificacion > 0:
                score += vecino_similitud * vecino_calificacion
                suma_similitud += vecino_similitud
        
        # Asignar puntaje promedio ponderado si hay similitud suficiente
        if suma_similitud > 0:
            recomendacion_scores[pelicula] = score / suma_similitud
    
    # Validar que existan recomendaciones
    if not recomendacion_scores:
        print("No se encontraron recomendaciones suficientes.")
        return pd.DataFrame(columns=['title'])
    
    # Ordenar y seleccionar las recomendaciones más altas
    peliculas_recomendadas = sorted(recomendacion_scores.items(), key=lambda x: x[1], reverse=True)
    recomendaciones_ids = [id for id, _ in peliculas_recomendadas[:num_recommendations]]
    recomendaciones = movies.set_index('movieId').loc[recomendaciones_ids][['title']]
    
    return recomendaciones


# # Presentar Recomendaciones para un Usuario específico:
# Se llama a la función de recomendación para un usuario específico, pasando el user_id y el número de recomendaciones que queremos obtener.

# In[23]:


#Ejemplo de recomendaciones para un usuario específico
user_id = 1  # Cambia el ID del usuario para probar con otros usuarios
print(f"Recomendaciones para el usuario {user_id}:")
print(recomendar_peliculas(user_id))


# In[ ]:





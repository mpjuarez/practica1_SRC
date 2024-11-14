#!/usr/bin/env python
# coding: utf-8

# # Sistema de Recomendación con KNN basado en ítem

# Librerías y carga de datos

# In[1]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Cargar el dataset de calificaciones y películas
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')


# # Matriz Usuario-Película:

# In[2]:


# Crear la matriz película-usuario y llenar NaN con ceros
movie_user_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Convertir la matriz a dispersa para mejorar la eficiencia
movie_user_sparse_matrix = csr_matrix(movie_user_matrix)


# # Algoritmo KNN para encontrar usuarios similares
# 
# Se utiliza el KNN para encontrar las películas más similares a cada una de las películas vistas.
# Usamos la métrica 'cosine' para calcular la similitud entre películas. 

# In[3]:


# Configurar el modelo KNN para similitudes entre ítems
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
knn.fit(movie_user_sparse_matrix)


# # Predicciones
# Predice la calificación que el usuario podría asignarle a la película.
# Para cada película similar, Si el usuario calificó la película, utiliza un promedio ponderado (similitud * calificación del usuario) y divide por la suma de similitudes para obtener la predicción.

# In[4]:


def predecir_calificacion_item_based(user_id, movie_id):
    # Verificar si el usuario ya calificó la película
    if movie_id in ratings[ratings['userId'] == user_id]['movieId'].values:
        print("El usuario ya ha calificado esta película.")
        return None
    
    # Obtener los ítems más similares a la película de interés
    movie_index = movie_user_matrix.index.get_loc(movie_id)
    distances, indices = knn.kneighbors(movie_user_sparse_matrix[movie_index], n_neighbors=10)
    
    # Calcular el promedio ponderado de las calificaciones para la predicción
    numerador, denominador = 0, 0
    for i in range(1, len(distances.flatten())):  # Evitar la misma película
        similar_movie_id = movie_user_matrix.index[indices.flatten()[i]]
        similarity_score = 1 - distances.flatten()[i]
        
        # Verificar si el usuario ha calificado esta película similar
        user_rating = movie_user_matrix.loc[similar_movie_id, user_id]
        if user_rating > 0:  # Solo considerar calificaciones existentes
            numerador += similarity_score * user_rating
            denominador += similarity_score
    
    # Calcular la calificación predicha si hay suficientes datos
    if denominador > 0:
        prediccion = numerador / denominador
        return prediccion
    else:
        # Retornar 0 en caso de no tener suficientes datos, para evitar errores en recomendaciones
        return 0


# # Recomendaciones
# Recomendar películas no vistas por el usuario.
# Para cada película en el sistema, si el usuario no la ha visto, la función calcula una predicción de calificación usando "predecir_calificacion_item_based".

# In[6]:


def recomendar_peliculas_item_based(user_id, num_recommendations=5):
    # Obtener las películas que el usuario ya ha calificado
    peliculas_vistas = ratings[ratings['userId'] == user_id]['movieId'].values
    
    # Diccionario para almacenar las predicciones de calificación para películas no vistas
    predicciones = {}
    
    # Iterar sobre todas las películas en el dataset
    for movie_id in movie_user_matrix.index:
        if movie_id not in peliculas_vistas:  # Considerar solo películas no vistas
            prediccion = predecir_calificacion_item_based(user_id, movie_id)
            # Agregar la predicción solo si es mayor que 0
            if prediccion > 0:
                predicciones[movie_id] = prediccion
    
    # Verificar si hay predicciones disponibles
    if not predicciones:
        print("No se encontraron recomendaciones suficientes.")
        return pd.DataFrame(columns=['title'])
    
    # Ordenar las películas no vistas por la calificación predicha más alta
    peliculas_recomendadas = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)
    recomendaciones_ids = [id for id, _ in peliculas_recomendadas[:num_recommendations]]
    
    # Obtener los títulos de las películas recomendadas
    recomendaciones = movies.set_index('movieId').loc[recomendaciones_ids][['title']]
    
    return recomendaciones

# Ejemplo de predicción de calificación y recomendación para un usuario específico
user_id = 3  # Cambia el ID del usuario para probar con otros usuarios
movie_id = 2  # Cambia el ID de la película para probar con otras películas

# Predicción de calificación para una película específica
prediccion = predecir_calificacion_item_based(user_id, movie_id)
if prediccion:
    print(f"Calificación predicha para el usuario {user_id} en la película {movie_id}: {prediccion:.2f}")
else:
    print("No fue posible predecir la calificación.")


# # Presentar recomendación

# In[7]:


# Recomendación de películas para el usuario
# Recomendación de películas para el usuario
print(f"Recomendaciones para el usuario {user_id}:")
print(recomendar_peliculas_item_based(user_id))


# In[ ]:





#PREGUNTA 5

pip install textdistance


import textdistance

def diferenciar_cadenas(cadena_1, cadena_2):
    distancia = textdistance.levenshtein.normalized_similarity(cadena_1, cadena_2)
    if distancia < 3:  # Establece un umbral para la diferencia
        return "Las cadenas son diferentes."
    else:
        return "Las cadenas son similares."

# Ejemplo de uso
cadena_1 = "Selena Leydi Tarqui"
cadena_2 = "Selena Leydi Tarqui M"

resultado = diferenciar_cadenas(cadena_1, cadena_2)
print(resultado)
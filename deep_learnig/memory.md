---
title: "CNN"
subtitle: "Transfer learning, fine tuning y linear probing"
author: "Ignacio Sánchez Herrera"
execute-dir: file
format: pdf
---

# Descripción de la arquitectura de CNN
La arquitectura escogida ha sido EfficienNet-B0.

Lo distintivo de EfficientNetB0 es su enfoque en el equilibrio entre el 
rendimiento y la eficiencia computacional. Logra esto mediante un proceso de 
escalado compuesto que aumenta simultáneamente la profundidad, el ancho y la 
resolución de la red de manera uniforme. Esto se realiza de manera eficiente 
utilizando un método de búsqueda de hiperparámetros llamado "compound scaling".

EfficientNetB0 consta de un bloque convolucional, seguido de 7 bloques 
denominados "MBConv", y finalmente tres capas de convolución, pooling y FC 
respectivamente. Los bloques MBConv representan un conjunto de capas entre las que se
encuentran convoluciones invertidas, funciones de activación y normalización, 
y reducción de dimensionalidad. Estos bloques utilizan operaciones de convolución de 
manera eficiente para reducir el número de parámetros y la carga computacional, 
manteniendo al mismo tiempo un buen rendimiento.

La arquitectura EfficientNetB0 también incluye técnicas como la regularización 
y el aumento de datos para mejorar la generalización y prevenir el sobreajuste, 
así como una función de activación SiLU.

# Descripción del conjunto de datos

EL conujunto de datos usado ha sido CIFAR-10.

CIFAR-10 contiene un total de 60,000 imágenes en color de tamaño 32x32 píxeles, 
divididas en 10 clases distintas, con 6,000 imágenes por clase. Estas clases 
son:

1. Avión
2. Automóvil
3. Pájaro
4. Gato
5. Ciervo
6. Perro
7. Rana
8. Caballo
9. Barco
10. Camión

# Resultados

Partiendo de la arquitectura de EfficientNetB0 se han creado dos modelos
diferentes. El primero consiste en realizar un Fine-tuning (FT) sin Transfer Learning (TL),
es decir, no se han usado los pesos de 'imagenet' sino que se han entrenado
los pesos desde cero. El segundo ha consistido en un Fine-tuning con Linear Probing (LP)
 y Transfer Learning.
 
Para una igualdad de condiciones los dos modelos se han entrenado durante el
mismo número de épocas. El modelo de Fine-tuning sin TL se ha entrenado
durante 5 épocas, mientras que el de TL con FT y LP se ha entrenado durante
3 épocas en la fase de LP y 2 épocas en la fase de FT.
 
Los resultados obtenidos se pueden ver en la siguiente tabla:

|**Modelo**|**T. entrenamiento**|**Accuracy en test**|
|:---:|:---:|:---:|
|FT sin TL|1234.46 sec|73.91%|
|TL, LP y FT| 772.58 sec | 92.71% |

Cómo vemos, las ventajas al realizar Transfer Learning con
Linear Probing y Fine Tuning son considerables, llegando a obtener
un 92% de *accuracy* (un ~19% más que sin TL) en el 62% del tiempo que tarda el 
modelo sin Transfer Learning. Seguramente se podría obtener un resultado mejor
en el primer modelo, pero esto requeriría entrenarlo durante muchas más épocas,
con el gasto de tiempo y capacidad de cómputo que esto conlleva.

Por lo tanto, podemos decir que el Transfer Learning es una muy buena opción
a la hora de usar modelos ya existentes y pre-entrenados ya que permiten
obtener muy buenos resultados con un tiempo de cómputo mucho menor.

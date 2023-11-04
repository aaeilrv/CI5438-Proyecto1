# Informe Proyecto I - CI5438

## 1 - Detalles de la implementación

Para hacer el algoritmo de descenso de gradiente, se utilizó la fórmula dada en clase para el caso multivariado con una tasa de aprendizaje igual a **0.000002**.

$$
h_w (x) = \sum_{i=1}^n w_i x_i
$$

A su vez, se hizo uso de la función de pérdida cuadrática $L^2(y,\hat{y}) = (y - \hat{y})^2$  tal como solicitó el enunciado. Para la condición de convergencia se seleccionó **500** como la cantidad máxima de iteraciones y un épsilon igual a **0.15**.

Se creó una función lineal de la forma $f(x) = w_1 x_1 + w_2 x_2 + w_0$ a manera de comprobar la funcionalidad del algoritmo. Para ella, se hizo uso de la librería *numpy* tanto para calcular los pesos iniciales como los atributos iniciales. Sin embargo, si se desea alterar el número de valores de la función lineal, se puede hacer a través del argumento *size* de *generate_linear_function()*, que se encarga de establecer el tamaño de la misma.

Dado que se trata de regresión linear multivariada, el valor $x_0$ para cada $w_0$ se tomó igual a 1, tal como fue explicado en clase. Por ende, los resultados obtenidos para $w_0$ no son tomados en cuenta al momento de conseguir una hipótesis.

Al generar el descenso de gradiente de $f(x)$, se obtuvieron pesos estimados con los cuales posteriormente se realizó una comparación con los pesos reales para obtener el margen de error de cada uno de ellos.

Se realizaron distintas corridas con $f(x)$. Además, se crearon otras funciones con mayor cantidad de coeficientes. A continuación, se muestra cada corrida con el tiempo de ejecución y el margen de error obtenido:

|                    | $X_2$ |$X_1$ |$X_0$ |
|  :----:            |:----: |:----:|:----:|
| Pesos reales       | -63   | 41   | -28  |
| Pesos estimados    |-63.35 |40.62 |0.714 |
| Porcentaje de error|0.56%  |0.92%|102.55%|

Tiempo de cálculo: 0.3s

![F1](/graficos/F1Lineal.png "F1")


|                    | $X_2$ |$X_1$ |$X_0$ |
|  :----:            |:----: |:----:|:----:|
| Pesos reales       | -32   | -4   | -14  |
| Pesos estimados    |-32.15 |-4.17 |0.56  |
| Porcentaje de error|0.49% |4.32%|104.04%|

Tiempo de cálculo: 0.3s

![F2](/graficos/F2lineal.png "F2")

|                    | $X_3$  |$X_2$ | $X_1$| $X_0$ |
|     :----:         | :----: |:----:|:----:| :----:|
| Pesos reales       |-91     |96    | 27   | -77   |
| Pesos estimados    |-89.83  |93.73 |26.92 | 0.77  |
| Porcentaje de error|  1.29% | 2.36%| 0.31%|  101% |

Tiempo de cálculo: 0.4s

![F3](/graficos/F3Lineal.png "F3")

|                    | $X_3$  |$X_2$ | $X_1$| $X_0$ |
| :----:             | :----: |:----:|:----:| :----:|
| Pesos reales       |99      |-98   |-24   |  50   |
| Pesos estimados    |99.41   |-97.76|-23.80| 1.39  |
| Porcentaje de error|  0.42% | 0.24%| 0.84%| 97.21% |

Tiempo de cálculo: 0.3s

![F4](/graficos/F4Lineal.png "F4")

|                    | $X_4$  |$X_3$ |$X_2$ | $X_1$ | $X_0$| 
| :----:             | :----: |:----:|:----:| :----:|:----:|
| Pesos reales       |  53    |-18   |16    |  -3   |  31  |
| Pesos estimados    |  43.50 |-11.26|-21.89| -2.25 | 0.81 |
| Porcentaje de error|  17.91%|37.46%|36.83%|   25% |97.37%|

Tiempo de cálculo: 0.4s

![F5](/graficos/F5Lineal.png "F5")

|                    | $X_4$  |$X_3$ |$X_2$ | $X_1$ | $X_0$| 
| :----:             | :----: |:----:|:----:| :----:|:----:|
| Pesos reales       |  97    | 72   | 22  |  42   |  1  |
| Pesos estimados    |  96.82 |72.05 |22.15| 41.89 | 2.19 |
| Porcentaje de error|  1.83% |7.10%| 6.86%| 2.45% |1.20%|

Tiempo de cálculo: 0.3s

![F6](/graficos/F6Lineal.png "F6")

Como se puede notar, el porcentaje de error de $X_0$ es muy elevado en todos los casos excepto la última función lineal, siendo esto producido por tomar $X_0 = 1$ para cada caso. Sin embargo, como ya fue explicado anteriormente, el valor estimado de este peso no es tomado en cuenta al momento de hallar una hipótesis para las funciones lineales multivariadas.

#### 1.1 Especificaciones técnicas
Los equipos utilizados para llevar a cabo este proyecto tienen las siguientes características:

1. Computador I:
- MD Ryzen 5 5600H with Radeon Graphics
- 8gb RAM
- Windows 11

2. Computador 2:
- Intel i3 11va generación
- 8gb RAM
- Windows 11

## 2 - Preprocesamiento de Datos
Se utilizó el conjunto de datos `CarDekho.csv` suministrado. En él, se encontraron los siguientes casos:
* Nueve columnas con valores nulos:

|     Columna      | Cantidad de valores nulos  |
|       :----:     |           :----:           |
|      Engine      |             80             |
|     Max Power    |             80             |
|    Max Torque    |             80             |
|    Drivetrain    |            136             |
|       Length     |             64             |
|        Width     |             64             |
|      Height      |             64             |
| Seating Capacity |             64             |
|Fuel Tank Capacity|            113             |

* Doce columnas con valores categóricos:
	* Make
	* Model
	* Fuel Type
	* Transmission
	* Location
	* Color
	* Owner
	* Seller Type
	* Engine
	* Max Power
	* Max Torque
	* Drivetrain
	
Procedemos a explicar el tipo de preprocesamiento que se utilizó para cada uno de estos casos.

### 2. 1 Columnas con valores faltantes:
En el conjunto de columnas con valores faltantes, se encuentran dos posibles tipos de valores:
* numéricos
* categóricos

#### 2. 1. 1 Numéricos:
Se decidió reemplazarlos por la mediana del atributo. Para esto, se hizo uso de las funciones *.loc[]* y *.mean()* de la librería *Pandas*:
*.loc[]* permite acceder a una columna del dataframe de acuerdo al valor que se le pase, mientras que *.mean()* obtiene la mediana de una columna de variables. Por ende, al hacer `df.loc[:, 'nombre_de_columna'].mean()` se obtiene la mediana de esa columna.

#### 2.1.2 Categóricos:
Se reemplazaron los valores vacíos por la moda del atributo. En este caso, se utilizó *value_counts()* de *Pandas* para obtener todas las categorías existentes en una columna y la cantidad de veces que se repetían, siendo aquella en la posición 0 la que se repite más. Así, al hacer `df.loc[:, 'nombre_de_columna'].value_counts().index[0]` se obtiene el valor más repetido de esa columna, es decir, su moda.

En ambos casos, con el uso de *.fillna()* -función también de *Pandas*- se obtienen los elementos nulos de cada columna y se rellenan con las medianas y modas obtenidas.

### 2.2 Manejo de valores categóricos
Se crearon $n$ nuevas columnas por las $n$ categorías existentes en cada atributo categórico.
Este procedimiento se hizo con ayuda de la librería *get_dummies* de Pandas, la cual  creó $n$ columnas numéricas para cada categoría.
Además de esto, nos vimos en la necesidad de crear una lista *indexes* que contiene en cada índice *i*, los *j* índices de las columnas *dummy* creadas en base a la columna *i* de la data original. Esto con motivo de poder hacer posteriormente una selección de modelo de manera más cómoda.

También se decidió eliminar las columnas *Max Power* y *Max Torque* vista la dificultad de estandarizar sus datos como valores numéricos.

### 2.3 Normalización
<sub>**Nota:** Este paso se realizó antes del manejo de valores categóricos. Sin embargo, se menciona de número 2.3 para seguir el hilo de la descripción dada en el enunciado del proyecto.</sub>

Después de haber completado los valores nulos existentes en el dataframe, se creó un boxplot para conocer el rango de valores de cada una de las columnas numéricas:

![Boxplot1](/graficos/Boxplot1.png "Boxplot1")

De ella, es notorio que las columnas *Price* y *Kilometer* tienen valores muy elevados con respecto a las otras columnas. Es por ende que se acudió a una normalización de todas las columnas numéricas para que sus valores estuviesen en el rango [0, 1]. Para ello, se creó una función *normalize_column()* que siguió las instrucciones dadas en el enunciado del proyecto:

$$
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

Se reemplazaron los valores numéricos de la siguiente manera:
`df.loc[:, 'nombre_columna'] = normalize_column(df.loc[:, 'nombre_columna'])`

Después de normalizados los valores, se creó otro boxplot:

![Boxplot2](/graficos/Boxplot2.png "Boxplot2")

Acá, se puede apreciar que todos los valores están en el rango deseado.

Después de la exhaustiva limpieza realizada, se decidió guardar el nuevo dataframe en un archivo llamado `clean_data.csv` para no alterar el dataframe original.

## 3 - Entrenamiento del modelo

### 3.1 - Selección del modelo

Para poder hallar un modelo que generara un margen de error pequeño al momento de obtener los precios estimados del vehículo, se decidió escoger todas las posibles combinaciones de 8 y 9 atributos del dataframe. Las razones para escoger estas cantidades fueron el límite de tiempo y la posibilidad de que el modelo estuviese subajustado con menos atributos. Las combinaciones obtenidas fueron guardadas en dos archivos de texto `c8.txt` y `c9.txt` respectivamente.

Luego, se definieron las funciones *model_selection()* y *cross_validation()*, esta segunda tomando los siguientes valores como parámetros:
* $k$ subconjuntos de partición
* archivo .txt con combinaciones
* dos índices $(a, b)$ de la lista de combinaciones para  ser tomados como intervalos.

La función *model_selection()* dividió a los ejemplos en dos partes: conjunto de entrenamiento y de prueba. Además, entrenó con validación cruzada el modelo con un subconjunto de combinaciones pasado como argumento, y finalmente hizo uso del algoritmo de descenso de gradiente.

Se estableció una matriz vectorial $X$ compuesta por los índices de todos atributos originales del dataframe con excepción de las columnas *Price*, *Max Power* y *Max Torque*. A su vez, se asignó $Y$ como el vector correspondiente a la columna *Price* del dataframe.

Finalmente, para escoger el modelo se probaron los siguientes intervalos:
* **[0, 1079] U [2500, 3120]** para las combinaciones de ocho elementos
* **[19540, 19640]** para las combinaciones de nueve elementos
Así como los siguientes parámetros para el descenso de gradiente:
* tasa de aprendizaje: **0.000002**
* máximo número de iteraciones: **500**
* epsilon: **0.15**

Para cada uno de estos intervalos, se obtuvo una mejor combinación encontrada que fue guardada manualmente en el archivo `best.txt`. Posteriormente, las combinaciones que se encontraban en este archivo volvieron a pasar por la selección de modelo pero con parámetros más estrictos:
* tasa de aprendizaje: **0.0000005**
* máximo número de iteraciones: **2000**
* epsilon: **0.15**

Así, finalmente la combinación escogida de atributos para calcular el precio de un vehículo fue la siguiente:
* Make
* Model
* Year
* Kilometer
* Fuel Type
* Transmission
* Drivetrain
* Width
* Seating Capacity

Con un total de 12.4% de margen de error.

### 3.2 - Selección de hipótesis
Finalmente, para seleccionar una hipótesis, se corrió nuevamente la función de descenso de gradiente, esta vez con $Y$ = *price* y $X$ con los atributos mencionados al final de la última sección.

Se corrieron varias sesiones de entrenamiento. Para la primera sesión los parámetros escogidos fueron los siguientes:
* tasa de aprendizaje: **0.0000015**
* máximo número de iteraciones: **900.000**
* epsilon: **0.01**

Para la primera sesión, se obtuvo a las 160.000 iteraciones, una convergencia con 0.9999% de margen de error. A continuación se muestra la gráfica de esta sesión.

![Grafica1](/graficos/Grafica1.jpg "Grafica1")

La sesión tomó 84min con 51 segundos.

La segunda sesión se corrió con los siguientes parámnetos:
* tasa de aprendizaje: **0.0000015**
* máximo número de iteraciones: **90.000**
* epsilon: **0.005**

A las 90.000 iteraciones se obtuvo un margen de error de 0.675%. A continuación se muestra la gráfica de esta sesión.

![Grafica2](/graficos/Grafica2.jpg "Grafica2")

La sesión tomó 53min con 28 segundos.

La última sesión de entrenamiento se realizó con los siguientes parámetros
* tasa de aprendizaje: **0.0000013**
* máximo número de iteraciones: **160.000**
* epsilon: **0.005**

A las 95.000 iteraciones se convergió a un margen de error de 0.4999%. A continuación se muestra la gráfica de esta sesión.

![Grafica3](/graficos/Grafica3.png "Grafica3")

La corrida tomó 48min con 11 segundos.

El entrenamiento en total tardó 186min con 30 segundos. Los pesos de la hipótesis final se encuentran en el archivo `w.txt`

## 5 - Conclusiones
En general, los resultados obtenidos al calcular el precio de vehículos fueron satisfactorios. Se estimaba un porcentaje más alto de error dado el margen de error de los atributos así como haciendo una comparación con el margen de error de las funciones lineales creadas al inicio del proyecto.
Para proyectos futuros o próximas visitas a este mismo, los autores de este proyecto plantean hacer los siguientes cambios a sus evaluaciones:
* Utilizar el mismo modelo de atributos cambiando la tasa de aprendizaje así como el epsilon, ajustándolos a valores tanto mayores como menores para ver si es posible obtener un menor margen de error.
* Utilizar distintos modelos, con más o menos atributos, pero manteniendo los valores de la tasa de aprendizaje y el epsilon, para comparar los márgenes de error con respecto al modelo utilizado.
* Separar los conjuntos de datos en conjuntos de entrenamiento y prueba con distintos porcentajes: 70% y 30%, 60% y 40%, o 50% cada uno.

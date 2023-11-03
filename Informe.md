
# Informe Proyecto I - CI5438

  

## 1 - Detalles de la implementación

Para hacer el algoritmo de descenso de gradiente, se utilizó la fórmula dada en clase para el caso multivariado con una tasa de aprendizaje igual a **0.000002**.

$$
h_w (x) = \sum_{i=1}^n w_i x_i
$$

A su vez, se hizo uso de la función de pérdida cuadrática $L^2(y,\hat{y}) = (y - \hat{y})^2$  tal como solicitó el enunciado. Para la condición de convergencia se seleccionó **500** como la cantidad máxima de iteraciones y un épsilon igual a **0.15**.

Se creó una función lineal de la forma $f(x) = w_1 x_1 + w_2 x_2 + w_0$ a manera de comprobar la funcionalidad del algoritmo. Para ella, se hizo uso de la librería *numpy* tanto para calcular los pesos iniciales como los atributos iniciales. Sin embargo, si se desea alterar el número de valores de la función lineal, se puede hacer a través del argumento *size* de *generate_linear_function()*, que se encarga de establecer el tamaño de la misma. Dado que se trata de regresión linear multivariada, el peso $w_0$ se considera no significativo para nuestro problema.

Al generar el descenso de gradiente de $f(x)$, se obtuvieron pesos estimados con los cuales posteriormente se realizó una comparación con los pesos reales para obtener el margen de error de cada uno de ellos.

Se realizaron distintas corridas con $f(x)$, además se crearon otras funciones con mayor cantidad de coeficientes. A continuación, se muestra cada corrida con el margen de error obtenido:

|                    | $X_2$ |$X_1$ |$X_0$ |
|  :----:            |:----: |:----:|:----:|
| Pesos reales       | -63   | 41   | -28  |
| Pesos estimados    |-63.35 |40.62 |0.714 |
| Porcentaje de error|0.56% |40.62%|102.55%|
(insertar imagen)

|                    | $X_2$ |$X_1$ |$X_0$ |
|  :----:            |:----: |:----:|:----:|
| Pesos reales       | -32   | -4   | -14  |
| Pesos estimados    |-32.15 |-4.17 |0.56  |
| Porcentaje de error|0.49% |4.32%|104.04%|
(insertar imagen)

|                    | $X_3$  |$X_2$ | $X_1$| $X_0$ |
|     :----:         | :----: |:----:|:----:| :----:|
| Pesos reales       |-91     |96    | 27   | -77   |
| Pesos estimados    |-89.83  |93.73 |26.92 | 0.77  |
| Porcentaje de error|  1.29% | 2.36%| 0.31%|  101% |
(insertar imagen)

|                    | $X_3$  |$X_2$ | $X_1$| $X_0$ |
| :----:             | :----: |:----:|:----:| :----:|
| Pesos reales       |99      |-98   |-24   |  50   |
| Pesos estimados    |99.41   |-97.76|-23.80| 1.39  |
| Porcentaje de error|  0.42% | 0.24%| 0.84%| 97.21% |
(insertar imagen)

|                    | $X_4$  |$X_3$ |$X_2$ | $X_1$ | $X_0$| 
| :----:             | :----: |:----:|:----:| :----:|:----:|
| Pesos reales       |  53    |-18   |16    |  -3   |  31  |
| Pesos estimados    |  43.50 |-11.26|-21.89| -2.25 | 0.81 |
| Porcentaje de error|  17.91%|37.46%|36.83%|   25% |97.37%|
(insertar imagen)

|                    | $X_4$  |$X_3$ |$X_2$ | $X_1$ | $X_0$| 
| :----:             | :----: |:----:|:----:| :----:|:----:|
| Pesos reales       |  88    | 44   | -76  |  38   |  95  |
| Pesos estimados    |  90.41 |39.55 |-74.04| 39.16 | 1.60 |
| Porcentaje de error|  2.73% |10.11%| 2.57%| 3.06% |98.32%|
(insertar imagen)

Como se puede notar, el porcentaje de error de $X_0$ es muy elevado en todos los casos, especialmente con respect a las otras $X_i$ de la misma corrida. Esto es causa de que, en la función *gradient_descent()* implementada, no se tuviese un trato distinto con este valor al momento de calcular la derivada parcial de $w$. En nuestro caso, optamos por ignorar el valor de $X_0$ al considerar la precisión de los pesos estimados ya que, como dijimos previamente, el mismo no es significativo al tratarse de funciones lineales multivaradas.  (VER SI EN VEZ DE ESTO SE PONE W_0 = 0 PARA TODO)

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
(IMAGEN BOXPLOT1)
De ella, es notorio que las columnas *Prices* y *Kilometer* tienen valores muy elevados con respecto a las otras columnas. Es por ende que se acudió a una normalización de todas las columnas numéricas para que sus valores estuviesen en el rango [0, 1]. Para ello, se creó una función *normalize_column()* que siguió las instrucciones dadas en el enunciado del proyecto:

$$
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

Se reemplazaron los valores numéricos de la siguiente manera:
`df.loc[:, 'nombre_columna'] = normalize_column(df.loc[:, 'nombre_columna'])`

Después de normalizados los valores, se creó otro boxplot:
(IMAGEN DE BOXPLOT2)

Acá, se puede apreciar que todos los valores están en el rango deseado.

## 3 - Entrenamiento del modelo
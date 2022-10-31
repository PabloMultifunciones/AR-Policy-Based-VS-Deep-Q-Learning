# AR-Policy-Based-VS-Deep-Q-Learning-
Aprendizaje Reforzado - Porque Policy Based es mejor que Deep Q Learning
### Introduccion ###

¿Por qué usar un algoritmo basado en políticas en lugar de Deep Q-learning?  

Una explicación súper simple sobre Policy Gradient.  

En Q-learning encontramos los valores Q como la suma esperada de recompensas dado un estado y una acción en el último método. Entonces, podemos usar un método tabular para almacenar todos los Q(s, a) o entrenar un aproximador como una red neuronal para mapear el estado y las acciones a los valores Q. Para elegir qué acción tomar dado un estado, tomamos la acción con el valor Q más alto (la recompensa futura máxima esperada que obtendré en cada estado).  

¡Así que Deep Q-learning es genial! ¿Por qué necesitamos otro método? Los científicos intentan encontrar otra forma de abordar los problemas de RL llamados basados en políticas. En este método, intentan encontrar la mejor política en un entorno en lugar de encontrar valores Q y luego actuar codiciosos.  

Los métodos basados en políticas tienen mejores propiedades de convergencia. Simplemente siguen un gradiente para encontrar los mejores parámetros, por lo que tenemos la garantía de converger en un máximo local (peor caso) o un máximo global (mejor caso). Además, los gradientes de política son más efectivos que los métodos tabulares. Mientras que la política concluye la acción, los métodos tabulares deben calcular los valores Q para todas las acciones. Imagina que tienes acción continua o tantas opciones para elegir.  

Una tercera ventaja es que los gradientes de política pueden aprender una política estocástica, mientras que las funciones de valor no. Significa que eliges entre acciones usando una distribución. Elija a1 con 40%, a2 con 20% y…. Por lo tanto, tiene un espacio de políticas más amplio para buscar.

Por ejemplo, imagina este pequeño entorno. En los bloques grises, debes ir a la derecha o a la izquierda. Cuando tiene una política determinista, nuestro agente se queda atascado. Pero en uno estocástico, el agente puede elegir derecha o izquierda dentro de una distribución. Por lo tanto, no se atascará y alcanzará el estado objetivo con una alta probabilidad.

![1_qYuzeb7XFh2gJZVt6X4SFA](https://user-images.githubusercontent.com/95035101/198917808-c3c86192-69ff-4598-89ce-8591e6ece29a.png)

![1_lpsBhxr5F52DUSYEAx5kaQ](https://user-images.githubusercontent.com/95035101/198917832-00bf068b-9e5f-4f5b-8162-782c42d0dec7.png)

Hasta ahora, entendemos otro tipo de algoritmo con algunos beneficios sobre el Q-learning profundo llamado gradiente de política, que sigue reglas de gradiente para encontrar parámetros que asignan el estado a la acción óptima.  

Entonces, ¿cómo debemos buscar en el espacio político? Nuestra elección es buena si maximiza la suma esperada de recompensas.

![1_bvmsDbBEr4u_xDGnt1fiGA](https://user-images.githubusercontent.com/95035101/198917932-f89c0120-0725-4356-b02d-e1e6776fad84.png)

Entonces, en entornos episódicos, la suma descontada de recompensas significa regresar desde el punto de partida. Imagine que siempre comienza desde s0 y luego la recompensa esperada de s0 usando esa política es su J. Puede reescribir la fórmula anterior como:

![1_Za4kHRrbHsfu2ZWYMxbUug](https://user-images.githubusercontent.com/95035101/198917968-232836e7-8971-450e-b02a-0171bb33d431.png)

Si no puede confiar en un estado de inicio específico, puede usar el valor promedio. Puede ponderar el promedio sobre V (s) para diferentes donde los pesos son la probabilidad de comenzar desde ese estado (o la probabilidad de que ocurra el estado respetado).


![1_ybKCWW_kvx7fqN9zzmEy5w](https://user-images.githubusercontent.com/95035101/198918011-08e09507-6787-44df-888a-adb8c1841f22.png)

Ahora puede reescribir V(s) como un promedio ponderado de las recompensas esperadas, donde los pesos son la probabilidad de elegir una acción específica.

![1_hhy_E-_XqejWC2gQwgn1MA](https://user-images.githubusercontent.com/95035101/198918049-b614dcae-1ed6-43b9-9e6e-2662852dc4a8.png)

Ahora que tenemos nuestra función objetivo, debemos usar el ascenso de gradiente (opuesto al descenso de gradiente) para maximizar J.

![1_2oj7Nc84TI3hYv6d7nLJIA](https://user-images.githubusercontent.com/95035101/198918069-dd46e840-9f66-4460-b789-ea440a1b9013.png)

Aquí necesitamos 2 lemas primero.

* Lemma 1:

![1_xyUxLJnz2Rl_uS43hgRJDQ](https://user-images.githubusercontent.com/95035101/198918112-5911e169-6cc1-481b-be3a-2ec704b48905.png)

* Lemma 2:
![1_kEPVajXZ6-1vcB33csfDmg](https://user-images.githubusercontent.com/95035101/198918131-bf8f764c-696c-45a6-86c9-ef9638847028.png)

Combine estos 2 lemas con nuestra función objetivo, podemos calcular el gradiente de J. Entonces, ahora el gradiente solo aplica nuestra política, que se puede modelar usando una red neuronal.

![1_MyX52fGNJ7fdEnVQ4m0R8w](https://user-images.githubusercontent.com/95035101/198918163-09cffdd2-4919-4e73-ad6a-8bfd08f68b89.png)

Escríbalo como una ecuación simple, nuestro enfoque de política de gradiente final se llama REINFORCE.

![1_lp2LTpNdkU2G1Mv4tp7R2Q](https://user-images.githubusercontent.com/95035101/198918193-198573fa-3e1a-4a61-a11c-00bb485196d5.png)

¡Aquí está el método de gradiente de política en la fórmula! Para terminar pongo el algoritmo del libro de Sutton:

![1_KjByUhq9oxfG3OMd0izUAQ](https://user-images.githubusercontent.com/95035101/198918236-36b4b765-13b2-49c9-ad81-c5129a1908bf.png)

uestro objetivo, por lo que debemos saber la recompensa acumulada al final del episodio. Es una especie de obedecer las reglas de Monte Carlo. Espere hasta que el agente finalice el episodio y luego cambie los parámetros y actualice la política. ¿Por qué esto es importante? Bueno, si haces una acción incorrecta en medio del episodio pero el episodio en general obtiene éxito, entonces crees que todas las acciones fueron lo suficientemente buenas. Significa que no puede reconocer si una acción afecta negativamente al episodio mientras ve los efectos generales. Entonces, tal vez en lugar de R, puede usar la recompensa esperada que puede obtener de ese estado y acción.

![1_JeqbiaaWkPaj6DA4jsLbKw](https://user-images.githubusercontent.com/95035101/198918266-016b1669-9634-432a-a6ce-256130c3c8f7.png)

Después de este cambio, ahora también debe estimar el valor Q. Es el segundo enfoque llamado Métodos críticos del actor. Cubriremos este tema en otra historia. Asegúrese de comprender el camino que recorremos paso a paso.



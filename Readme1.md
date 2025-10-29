#### Installer les dépendances , les bibliotheques tensorflow numpy matplotlib pandas seaborn:
via l'instruction
!pip install tensorflow numpy matplotlib pandas seaborn jupyter
## Objectif
Notre projet compare deux architectures de réseaux de neurones pour la classification des chiffres manuscrits du dataset **MNIST** :
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

## Étapes principales
1. Prétraitement des données (normalisation, reshape, one-hot encoding)
2. Entraînement de deux modèles distincts
3. Évaluation et comparaison sur le jeu de test
4. Visualisation des résultats (accuracy, loss, temps)

### Prétraitement des données MNIST

### Une fonction pour charger le dataset **MNIST** et effectuer le prétraitement.


```python
def load_and_preprocess_data(for_lstm=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalisation
    
    if for_lstm:
        x_train = x_train.reshape(x_train.shape[0], 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
```
## Implémentation du CNN:

```python
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data(for_lstm=False)


cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```
## Tableau de CNN:
![Resultat_CNN](cnn_resultat.PNG)

### Implémentation du LSTM.
```python
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data(for_lstm=True)

model = Sequential([
    LSTM(128, input_shape=(28,28)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```
## Interprétation des résultats
 1. Précision (Accuracy)
Le CNN obtient une meilleure précision (99.19 %) que le LSTM (98.55 %).
Cela s’explique par la nature spatiale des images MNIST :
Le CNN exploite les corrélations locales entre pixels grâce aux filtres de convolution.
Le LSTM, conçu pour traiter des données séquentielles (textes, séries temporelles), n’est pas naturellement adapté aux images 2D.
Pour le LSTM, chaque ligne ou pixel est traité comme une séquence, donc il perd de l’information spatiale importante.

![Précision Accuracy](comparaison%20accuracy.PNG)

2.Temps d’entraînement
 
Le LSTM est plus lent (207 s) que le CNN (184 s).
Les LSTM traitent les données séquentiellement, donc ils ne peuvent pas paralléliser aussi efficacement que les convolutions.
Le CNN utilise des opérations matricielles hautement optimisées.
![Temps d’entraînement](comparaison%20tems%20d'entrainement.PNG)
##  Discussion
Le CNN atteint une meilleure précision car il capture efficacement les motifs spatiaux des images.
Le LSTM, conçu pour des données séquentielles, reste performant mais moins adapté.
Interprétation des résultats

## Discussion 

Sur le dataset MNIST, le modèle CNN surpasse le modèle LSTM à la fois en précision et en efficacité temporelle. Cela s’explique par la nature des images : les CNN sont spécialement conçus pour capturer les dépendances spatiales locales grâce aux couches de convolution et de pooling, tandis que les LSTM sont destinés à modéliser des séquences temporelles. Ainsi, bien que le LSTM puisse atteindre une précision correcte, il est moins efficace pour ce type de données et demande un temps d’entraînement plus long.

En perspective, le LSTM pourrait être plus intéressant sur des données séquentielles (par ex. reconnaissance de gestes, séries temporelles d’images ou texte manuscrit continu), tandis que le CNN reste la référence pour la classification d’images fixes comme MNIST.

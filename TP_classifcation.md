# Sujet : Classification d'images avec le jeu de données CIFAR-10


## Contexte
Vous allez travailler sur le jeu de données CIFAR-10, qui contient 60 000 images de 10 classes différentes (aéronefs, automobiles, oiseaux, chats, cerfs, chiens, grenouilles, chevaux, navires, camions). L'objectif de ce projet est de tester divers modèles de réseaux de neurones pour la classification d'images et d'explorer l'influence des hyperparamètres sur les performances du modèle.

## Objectif
L'objectif principal est de tester les réseaux de neurones que je vous propose et d'analyser leurs performances. Vous aurez à explorer les différents modèles et les hyperparamètres d'apprentissage (tels que le taux d'apprentissage, le nombre d'époques, la taille des lots, et d'autres variables) . Vous utiliserez des fonctions de perte et de précision pour évaluer vos modèles.

## Tâches à réaliser
1. **Utilisation du notebook d'exemple :** Vous commencerez par un notebook d'exemple fourni [CIFAR10_classification.ipynb](classification/CIFAR10_classification.ipynb), qui contient des modèles minimaux à tester. L'idée sera de le faire évoluer pour l'inclure d'une maniere ou d'une autre dans votre rendu. L'idéal est de l'exécuter sous Google Colab avec utilisation d'un GPU (ex. T4).


| :boom: DANGER              |
|:---------------------------|
| :exclamation: Deconnecter vous de votre  session d'execution quand vous ne faites pas de calcul sous risque de vous faire bannir temporemement de l'usage des GPUs  :exclamation:|

2. **Expérimentations avec les modèles :** Vous devrez tester les modèles proposés dans le notebook et ajuster les hyperparamètres pour voir comment cela impacte les performances.

 
 Au passage il faudra bien sur expliquer tous les parametres utilisés pour l'apprentissage . Il est aussi important de définir les critéres de performances . Vous expliciterez en particulier pourquoi les plus importants sont associés au jeux de données de test (par oppposition à celui d'apprentissage) 

3. **Analyse des couches :** Dans votre notebook, vous devrez expliquer en details  les différentes couches utilisées dans les modèles (couches denses, couches de convolution, etc.) et leur rôle dans le traitement des données. Citer vos references et vos  sources en particulier d'illustration de la convolution etc ... A la lecture de cette partie on doit comprendre le principe des differentes couches.

> [!IMPORTANT]
> La compréhension de cette partie  est l'interet principal de vos activités

4. **Proposition d'améliorations :** Vous êtes invités à proposer des solutions pour obtenir le meilleur score possible. Si vous avez utilisé des ressources extérieures, veillez à les citer dans votre travail.

5. **Synthèse des résultats :** Votre rendu final devra inclure des tableaux de synthèse des performances des différents modèles et des courbes montrant l'influence des choix effectués.

6. **Explicabilité :** Une derniere partie s'interessera à l'explicabilité des résultats , en utilisant par exemple GradCAM++ implémenté sous [tf_keras_vis](https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#GradCAM++). Vous justifirez ou critiquerez mon choix. Proposez autres choses si ca vous semble plu pertinent . Dans tous les cas, on appliquera au modele et aux images utilisés dans le notebook.

7. **Specialisation des couches (BONUS)** On aimerait voir comment se sont spécialiser les couches ? Comment représenter cela ? Proposez une solution pour les voir
   
8.  **Application sur d'autres dataset**  Tester vos reseaux sur d'autres dataset existants en procédant à un nouvel apprentissage (on peut par exemple tester sur la datset MNIST , MNIST Fashion, essayer de trouver 2 autres dataset) Afficher les résultats obtenus (de manière synthétiques  et Montrer si la spécilisation a été différentes)

9. **Enrichissement du dataset CIFAR10** Créer 4 images par classe du CIFAR10 pour enrichier le dataset CIFAR utilisé au début (Montrer precismeent comment vous avez procéder et quelle était l'image de départ, quelle sont les resultats obtenus avec le reseau precedemment entrainé ? Sont ils satisfaisant ? Quelle est la taille d'origine du dataset CIFAR10 ?

10. **Enrichissement du dataset Fashion MNIST** Même questions mais avec  juste 2 images pour 3 classes. Bien entendu tester des modèles après les avoir créer ... 

## Rendu
- Le rendu doit être sous gitlab
- L'idéal est de d'inclure en plus des markdowns habituels un ensmeble de fichiers Jupyter Notebook, hébergé sur Gitlab est associé à votre rendu complet .
- L'avantage du format ipynb et qu'il contient le résulat de votre derniere execution avant sauvegarde mais il peut être aussi complétement regenerer avec les codes pythons
- Assurez-vous que votre notebook est bien documenté et que chaque section est clairement indiquée. Vous pouvez rédiger en francais ou en anglais 

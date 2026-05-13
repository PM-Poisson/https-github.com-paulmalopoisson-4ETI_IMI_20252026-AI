# TP : Detection par Deep Learning - Mise en œuvre de Yolo

## Sujet

**Prérequis** :  
Connaissances à minima du traitement d'images. Bases d'algorithmie. Connaissances à minima de python. Prise en main d'OpenCV.

## Introduction
De nombreuses techniques peuvent être utilisées pour aider à la reconnaissance d'objets et des activités associées (classification, labellisation, détourage, description, extraction, manipulation des images…). Le Deep Learning est l'une des approches les plus efficaces du moment. On peut voir sur le schéma suivant que le Deep Learning est une branche du Machine Learning, bio-inspiré par le fonctionnement du cerveau.

La particularité du Deep Learning réside dans les techniques de mise à jour des poids du réseau par des techniques dites de backpropagation. Elles se révèlent très simples à mettre en œuvre mais nécessitent de nombreuses phases d'itération et un dataset important et bien choisi pour converger vers des solutions efficientes.

Dans le cas du Deep Learning, le nombre de neurones peut être très important, organisé sous forme de couches pour limiter les interactions. Les couches les plus proches des entrées garantissent un traitement identique pour l'ensemble des pixels. Cette structure se spécialise de plus en plus en remontant dans les couches, avec des couches finales ultra-spécialisées. Cette propriété permet de supprimer la phase de création de modélisation et d'analyse des "features" (caractéristiques), traditionnellement créée par des êtres humains.

De nombreux outils de gestion de réseaux de neurones existent, et de nombreuses applications sont mises en place. Les concepteurs doivent développer ou choisir une architecture (nombre de neurones, organisations de couches, mécanismes d'apprentissage) puis créer le dataset nécessaire à l'apprentissage. Finalement, le réseau peut être figé une fois que les performances sont jugées correctes.

Une des  solutions les plus populaire actuellement est **TensorFlow** (on trouve aussi pytorch), qui fournit un cadre simple pour le développement dans tous les domaines. Le module suivant "Deep Learning" sera centré sur TensorFlow. La solution choisie pour notre TP est le réseau **YOLO**, utilisé dans différents frameworks. La particularité de YOLO est son analyse d'image en une seule passe, détectant des objets à différentes échelles.

[Réalisation en YOLO](https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

Le rendu sera fait sur git avec un rapport contenant les codes et exemples
---

## Partie 1 : Prise en main de YOLO dans le module DNN d'OpenCV

1. **Question 1.1** : Quelles sont les différentes classes reconnues par le réseau de neuronne  ?  Tester les réseaux Tiny-YOLO (disponible dans detection [detection](detection)) et YOLO (à télécharger [YOLO](https://pjreddie.com/darknet/yolo/) sur une dizaine d'images ([test_opencv_dnn_yolo_coco.py](detection/test_opencv_dnn_yolo_coco.py))

80 classes (liste dans coco.names)

2. **Question 1.2** : À quoi sert le ou les thresholds ?

ignorer les objets pas sûrs (<50%)
éviter les doublons de boites sur les mêmes objets 

3. **Question 1.3** : Quels sont les fichiers utilisés liés au réseau de neurones, que contiennent-ils précisément ?

yolov3-tiny.cfg :architecture du réseau
.weights poids
coco.names noms des classes 

4. **Question 1.4** : Quelle est l’architecture du réseau YOLOv3 et YOLOv3 tiny ? Expliquer les différentes couches du réseau (pas la peine de rentrer trop dasn le detail, cela fera l'objet d'une autre activitée). Lesquelles dépendent du nombre de classes ?

convolution activation yolo 
5. **Question 1.5** : Trouvez-vous d’autres modèles pré-entraînés disponibles au téléchargement ? Tester les.
oui yolo darknet 
---

## Partie 2 : Interfaçage Python

1. **Question 2.1** : Ajouter une option pour n'afficher qu'une liste de classe configurée en ligne de commande. Adapter le code pour fonctionner sur un flux webcam ou des images. Inclure le code commenté.
2. **Question 2.2** : Générer un fichier JSON contenant des informations sur les classes, les bounding boxes, et le flux/image de départ. Inclure le code et montrer le résultat.
3. **Question 2.3** : Sauvegarder le contenu des bounding boxes sous forme d’images JPEG. Générer une mosaïque d’images pour chaque classe. Inclure le code et montrer le résultat.

---

## Partie 3 : Préparation à l’entraînement de votre propre classificateur

Choissisez entre 5 et 10 objets, vous devrez faire une vingtaine de photo par objets, ideallement avec des fonds différents et plusieurs objets par photo et des objets que vous ne voulez pas reconnaitre. 

1. **Question 3.1** : Utiliser [**Roboflow**](https://roboflow.com/) pour créer un dataset (detector, bounding box ), puis lancer l’apprentissage sur Roboflow. Tester les résultats. Comment récupérer et utiliser le réseau généré ?
2. **Question 3.2** :Exporter au format ultralytics Hub (download dataset puis format hultralytics hub).Rq :   Sous [**Ultralytics**](https://hub.ultralytics.com/) Lancer l'entrainement sous google collab en utilisant le format **YOLOv5lu**. Tester les résultats et comparer avec ceux de Roboflow. Vous pouvez exporter ensuite au format **ONNX**, 
> [!CAUTION]
> Si la procedure d'export automatique de **Roboflow** vers **Ultralytics**  ne fonctionne pas, exporter au format yolov5 pytorch sur votre ordinateur pour le reinjecter comme dataset sous **Ultralytics**.
hors opencv(dnn) 
4. **Question 3.3** : Repartir du code ([opencv_onxx_example.py ](detection/opencv_onxx_example.py ))  et l’adapter pour utiliser votre réseau ONNX. Expliquer les modifications effectuées, inclure le code et les captures d’écran.

> [!IMPORTANT]
> Malheuresuement les versions d'openCV sont ne fonctionne pas toujours en fonction des formats ONXX . Vous devrez peut etre tester le format original pytorch sans passer par opencv (DNN). Par exemple avec la librairie ultralytics)
   

---

## Bibliographie complémentaire

- Outils de labellisation :  
  [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)  
  [OpenLabeling](https://github.com/fabricejumel/OpenLabeling)

- Sur YOLO :  
  [YOLOv3 en PyTorch](https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html)  
  [YOLOv3 spécificités](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

- Divers :  
  [Yolo-opencv](https://datacorner.fr/yolo-opencv/)  
  [OpenCV YOLO tutorial](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)


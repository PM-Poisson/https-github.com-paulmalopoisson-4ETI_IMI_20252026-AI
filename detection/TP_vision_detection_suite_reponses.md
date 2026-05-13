# Sujet : Détection d'Objets avec TensorFlow — Réponses

---

## 1. Question 1 (Rappel)

### 1.a. Différence entre classification, détection et segmentation

- **Classification** : le réseau prédit *ce qu'il y a* dans l'image, sans localiser. Il retourne un label global (ex : "chat"). Pas de localisation.

- **Détection** : le réseau prédit *ce qu'il y a ET où c'est*. Il retourne des **bounding boxes** rectangulaires autour de chaque objet, avec un label et un score de confiance. C'est ce que fait YOLO.

- **Segmentation** : plus précise que la détection. Elle prédit la localisation au niveau du **pixel**. En segmentation *sémantique*, chaque pixel reçoit un label de classe. En segmentation *d'instances*, chaque objet individuel est délimité par un contour exact (masque polygonal).

### 1.b. Grandes solutions de détection d'objets

- **YOLO** (You Only Look Once) : détection en une seule passe du réseau, très rapide, adapté au temps réel. Versions : YOLOv3, v5, v8, v11, v26…
- **SSD** (Single Shot MultiBox Detector) : détection en une passe également, utilise des feature maps à différentes échelles.
- **Faster R-CNN** : détecteur deux étapes — un réseau RPN propose des régions, puis un réseau classifie chacune. Plus précis mais plus lent.
- **RetinaNet** : détecteur une étape avec la Focal Loss pour mieux gérer les déséquilibres de classes.
- **EfficientDet** : architecture scalable combinant EfficientNet comme backbone et BiFPN comme feature network.
- **DETR** (Detection Transformer) : approche basée sur les Transformers, sans ancres ni NMS.

---

## 2. Question 2

*(Basé sur l'exemple TensorFlow Hub : tf2_object_detection.ipynb)*

### 2.a. Classes reconnues par le réseau

Les modèles de l'exemple sont entraînés sur le dataset **COCO** (Common Objects in Context) qui contient **90 classes** : personnes, véhicules (voiture, bus, moto, vélo, avion, bateau…), animaux (chat, chien, cheval, vache, oiseau…), objets du quotidien (chaise, table, bouteille, téléphone, ordinateur portable, fourchette…) et plus.

### 2.b. Chargement du modèle — partie du code

Le chargement se fait via TensorFlow Hub avec la ligne :
```python
hub_model = hub.load(model_handle)
```
où `model_handle` est une URL pointant vers un modèle hébergé sur TensorFlow Hub (ex : `https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2`).

Les modèles proposés dans l'exemple incluent notamment :
- `ssd_mobilenet_v2` — léger et rapide
- `efficientdet_d0` à `d7` — plus précis, scalable
- `faster_rcnn_inception_resnet_v2` — très précis, plus lent
- `centernet_hourglass104` — architecture Hourglass basée sur CenterNet

### 2.c. Structures des modèles sous-jacents

- **SSD MobileNetV2** : backbone MobileNetV2 (couches dépthwise separable convolutions) + tête SSD multi-échelle. Très léger, conçu pour le mobile.
- **EfficientDet** : backbone EfficientNet + BiFPN (Bidirectional Feature Pyramid Network) pour la fusion de features multi-échelles.
- **Faster R-CNN / Inception ResNet** : backbone Inception-ResNet-v2 + Region Proposal Network (RPN) + tête de classification. Architecture deux étapes, très précise.
- **CenterNet Hourglass** : détecte les objets comme des points centraux (keypoints), backbone Hourglass Network (empilement de modules encoder-decoder).

### 2.d. Tableau comparatif sur images tests

| Image | SSD MobileNetV2 | EfficientDet-D4 | Faster R-CNN |
|---|---|---|---|
| Rue animée | Personnes ✓, voitures ✓ | Personnes ✓, voitures ✓, vélos ✓ | Personnes ✓, voitures ✓, feux ✓ |
| Cuisine | Bouteille ✓, tasse ~ | Bouteille ✓, tasse ✓, fourchette ✓ | Bouteille ✓, tasse ✓, fourchette ✓ |
| Animal | Chien ✓ | Chien ✓ | Chien ✓ |
| Sport | Personne ✓ | Personne ✓, ballon ✓ | Personne ✓, ballon ✓ |
| **Vitesse** | Rapide | Moyenne | Lente |
| **Précision** | Correcte | Bonne | Très bonne |

*(Résultats indicatifs — à compléter avec tes propres captures d'écran)*

---

## 3. Question 3

### 3.a. À quoi servait TensorFlow Hub ? Quels sont ses remplaçants ?

**TensorFlow Hub** était une plateforme de Google permettant de partager et réutiliser des modèles de machine learning pré-entraînés (embeddings, détecteurs, classificateurs…). On pouvait charger un modèle en une ligne avec `hub.load(url)`.

Depuis, TF Hub est progressivement remplacé par :
- **Kaggle Models** (https://kaggle.com/models) — nouveau dépôt officiel Google pour les modèles TF/Keras
- **Hugging Face Hub** (https://huggingface.co/models) — plateforme multi-framework (PyTorch, TF, JAX) très populaire
- **Ultralytics Hub** (https://hub.ultralytics.com) — spécialisé YOLO
- **ONNX Model Zoo** (https://github.com/onnx/models) — modèles au format ONNX

### 3.b. Nombre de réseaux de détection d'objets

- **Hugging Face** : plusieurs centaines de modèles de détection d'objets (tâche `object-detection`)
- **Kaggle Models** : dizaines de modèles de détection (SSD, EfficientDet, Faster R-CNN…)
- **Ultralytics Hub** : famille YOLO (v5, v8, v11, v26, nano à xlarge) soit une vingtaine de variantes

### 3.c. Architectures des réseaux

On retrouve principalement :
- **Architectures une étape** : YOLO (CSPDarknet, backbone + neck FPN/PAN + tête de détection), SSD (MobileNet/VGG + multi-scale heads)
- **Architectures deux étapes** : Faster R-CNN (backbone + RPN + RoI pooling + classifieur)
- **Architectures Transformer** : DETR (encoder-decoder Transformer + backbone ResNet), DINO, Co-DETR
- **Architectures Hourglass/CenterNet** : détection par keypoints

### 3.d. Classes reconnues

La plupart des modèles généralistes sont entraînés sur :
- **COCO** : 80 à 90 classes (objets courants, animaux, véhicules)
- **ImageNet** : 1000 classes (pour la classification, base du pré-entraînement)
- **Objects365** : 365 classes (plus varié que COCO)
- **LVIS** : 1200+ classes (longue traîne d'objets rares)

### 3.e. Exemples de fine-tuning

Oui, les trois plateformes proposent des exemples :
- **Hugging Face** : tutoriels de fine-tuning avec `transformers` et `datasets` pour DETR, DETA…
- **Ultralytics** : documentation complète pour fine-tuner YOLOv8/v11 sur dataset custom avec `model.train(data="monDataset.yaml")`
- **Roboflow** : fine-tuning intégré dans l'interface web, export vers Ultralytics ou ONNX

---

## 4. Question 4

### 4.a. Format d'entrée d'un réseau de détection

Un réseau de détection attend en entrée un **tensor 4D** de forme `[batch, canaux, hauteur, largeur]`.

Exemple avec **YOLOv8n** :
- Taille d'entrée : `[1, 3, 640, 640]` (1 image, 3 canaux RGB, 640×640 pixels)
- Valeurs normalisées entre 0 et 1 (division par 255)
- En sortie : tensor `[1, 84, 8400]` — 84 = 4 coords bbox + 80 scores de classes, pour 8400 ancres candidates

### 4.b. Transformer un classifieur en détecteur

Un classifieur produit un seul vecteur de probabilités pour toute l'image. Pour en faire un détecteur, on peut :

1. **Retirer la couche fully-connected finale** du classifieur (ex : ResNet50)
2. **Ajouter une tête de détection** : couches convolutives qui prédisent, pour chaque position spatiale, les coordonnées d'une bounding box et les scores de classe
3. **Ajouter un mécanisme multi-échelle** (FPN — Feature Pyramid Network) pour détecter des objets de tailles différentes

Exemple avec **ResNet50 → Faster R-CNN** :
- On garde ResNet50 comme backbone (feature extractor)
- On ajoute un RPN (Region Proposal Network) sur les feature maps
- On ajoute une tête de classification par région (RoI Pooling + FC)

### 4.c. Phase d'apprentissage dans ce cas

La phase d'apprentissage nécessite :
- Des **images annotées avec des bounding boxes** (pas juste des labels globaux)
- Une **fonction de perte combinée** : perte de localisation (régression des coords bbox, ex : MSE ou IoU Loss) + perte de classification (cross-entropy)
- Un **dataset annoté** au format approprié (COCO JSON, YOLO txt, Pascal VOC XML…)
- Potentiellement un **scheduler de learning rate** et des techniques d'augmentation (flip, crop, mosaic…)

### 4.d. Intégrer un classifieur pré-entraîné pour en faire un détecteur

Oui, c'est la technique du **transfer learning**. Le backbone pré-entraîné (ex : ResNet, VGG, EfficientNet) sert d'extracteur de features. Seule la tête de détection est initialisée aléatoirement. Cela permet de bénéficier des features apprises sur ImageNet (formes, textures, contours) sans repartir de zéro.

### 4.e. Fine-tuning — procédure et format des données

Oui, il faut procéder à un **fine-tuning**, avec deux stratégies :

- **Fine-tuning partiel** : on gèle le backbone et on entraîne uniquement la tête de détection (plus rapide, moins de risque d'overfitting sur petit dataset)
- **Fine-tuning complet** : on entraîne tout le réseau avec un learning rate très faible pour le backbone (plus lent, meilleur si dataset suffisamment grand)

**Format du jeu de données** :
- Chaque image doit être accompagnée d'annotations de bounding boxes
- Formats courants : **YOLO txt** (une ligne par objet : `class cx cy w h` normalisés), **COCO JSON**, **Pascal VOC XML**
- Répartition en train / validation / test (typiquement 70/20/10)

### 4.f. (Bonus) Créer un détecteur depuis un CNN vanilla

*(Non traité en priorité comme indiqué dans le sujet)*

---

## 5. Question 5

### 5.a. kitt.tools/ai/image-recognition

Le site **kitt.tools** propose un outil de reconnaissance d'images accessible directement dans le navigateur, sans installation. Il tourne entièrement côté client grâce à des modèles optimisés au format **ONNX** exécutés dans des Web Workers.

### 5.b. Solutions mises en œuvre

Le site implémente une approche **multi-modèles**, avec un modèle spécialisé par tâche :
- **Détection d'objets généraux** : DETR-ResNet50
- **Classification d'images** : ViT-Base (Vision Transformer)
- **Reconnaissance de texte** : TrOCR
- **Compréhension de scène** : CLIP (Contrastive Language-Image Pre-Training)

### 5.c. Architectures sous-jacentes

- **DETR-ResNet50** : Transformer encoder-decoder avec backbone ResNet50 — prédit directement un ensemble fixe de détections sans NMS
- **ViT-Base** : Vision Transformer — découpe l'image en patches 16×16, les encode comme des tokens et utilise l'attention multi-têtes pour la classification
- **TrOCR** : Transformer encoder (ViT) + decoder (BERT-like) pour la reconnaissance de texte dans les images
- **CLIP** : deux encodeurs Transformer (un pour les images, un pour le texte) entraînés conjointement pour aligner images et descriptions textuelles

### 5.d. Classes reconnues

- **DETR** : 91 classes COCO (objets courants)
- **ViT-Base** : 1000 classes ImageNet (très large spectre)
- **CLIP** : pas de classes fixes — fonctionne en **zero-shot** en comparant l'image à n'importe quelle description textuelle
- **TrOCR** : caractères et mots (pas de classes objet)

### 5.e. Autres usages de ces modèles

- **CLIP** : recherche d'images par texte, génération d'images guidée (Stable Diffusion l'utilise), zero-shot classification, similarité image-texte
- **ViT** : feature extraction pour d'autres tâches vision, segmentation (ViT-SAM), médical imaging
- **TrOCR** : numérisation de documents, lecture de plaques, sous-titrage automatique
- **DETR** : détection panoptique (objets + fond), point de départ pour des variantes (Deformable DETR, DINO…)

### 5.f. Générer des bounding boxes avec ces modèles

- **DETR** : oui, directement — il génère nativement des bounding boxes (c'est un détecteur)
- **CLIP** : pas directement, mais on peut combiner CLIP avec une technique de **Grad-CAM** ou utiliser **CLIP + SAM** (Segment Anything Model) pour générer des boîtes à partir d'une description textuelle
- **ViT** : non directement, mais on peut utiliser l'attention pour localiser grossièrement les régions importantes (attention rollout)

Exemple avec DETR en Python :
```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("photo.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-traitement : récupération des bounding boxes
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"{model.config.id2label[label.item()]} {score:.2f} — box: {box.tolist()}")
```

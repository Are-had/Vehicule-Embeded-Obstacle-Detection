# Projet de Detection d Obstacles pour Vehicules Autonomes

Ce depot regroupe les travaux de developpement lies a la detection et a la classification d obstacles routiers via l architecture YOLOv8.

## Etat de l Analyse

Le projet fait actuellement l objet d une analyse plus approfondie sur serveur haute performance (GPU). Cette phase permet de traiter des volumes de donnees plus importants et d affiner la precision des poids. Des modeles plus performants, optimises pour la detection de petits objets et la reduction des faux positifs, sont en cours de deploiement.

## Organisation du Depot

- datasets/ : Fichiers de configuration YAML pour la gestion des donnees (BDD100K, Road Damage, Lost and Found).
- detection-and-classification/ :
    - training/ : Scripts d entrainement pour les modeles YOLOv8 (Nano et Large).
    - inference/ : Notebooks de test pour la validation sur flux reels.
    - evaluation/ : Analyse detaillee des metriques (mAP, matrices de confusion) et dossier samples contenant les echantillons de resultats.
    - utils/ : Scripts de conversion de formats d annotations et outils de pretraitement.

## Objectifs Principaux

1. Analyse et detection de debris routiers.
2. Reconnaissance de la signalisation (Stop, Travaux).
3. Identification des anomalies de la chaussee (Dos d ane).
4. Benchmarking des performances sur serveur pour une future integration embarquee.

## Installation

Pour configurer l environnement :
pip install -r requirements.txt

Les donnees de sortie et les preuves visuelles de performance sont repertoriees dans le dossier evaluation/samples/.

# Deep Learning — Comparaison d'approches pour la restauration d'images

Ce projet compare trois architectures de deep learning pour la restauration d'images bruitées : une approche déterministe supervisée (U-Net), une approche générative supervisée (VAE) et une approche self-supervisée (Noise2Void). Il a été réalisé dans un cadre pédagogique afin d'analyser les forces et limites de chaque méthode.

- **U-Net** : modèle déterministe entraîné en apprentissage supervisé sur des paires (image bruitée, image propre). Il apprend à inverser directement la dégradation.
- **VAE** (Variational Autoencoder) : modèle génératif qui apprend une représentation latente continue et structurée. Il peut débruiter en projetant l'image dans son espace latent et offre une meilleure robustesse hors distribution grâce à la régularisation KL.
- **Noise2Void** : modèle self-supervisé qui apprend uniquement à partir d'images bruitées, sans jamais accéder à une image propre. Il prédit des pixels masqués à partir de leurs voisins, ce qui lui permet de généraliser à tout bruit spatialement indépendant.

Les modèles sont évalués avec les métriques PSNR, SSIM et LPIPS, sur différents niveaux de bruit gaussien ainsi que sur des dégradations jamais vues à l'entraînement (bruit fort, sel et poivre, flou gaussien).

## Structure du projet
DEEP-LEARNING/
├── eval_results/          – Résultats des évaluations (métriques, logs)
├── graphs_figures/        – Graphiques et figures générés
├── logs/                  – Fichiers de log d'entraînement
├── models/                – Modèles sauvegardés (checkpoints)
├── notebooks_analysis/    – Notebooks d'analyse comparative
├── dataset.py             – Chargement et prétraitement des données
├── metrics.py             – Métriques (PSNR, SSIM, LPIPS)
├── creer_bruit.py         – Génération de bruit sur les images
├── train_unet.py          – Entraînement du U-Net
├── train_vae.py           – Entraînement du VAE
├── train_n2v.py           – Entraînement de Noise2Void
├── evaluate_models.py     – Évaluation comparative des modèles
├── evaluate_vae.py        – Évaluation spécifique du VAE
├── main.py                – Script principal
├── requirements.txt       – Dépendances Python
└── README.md

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
.\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Entraînement des modèles

```bash
# U-Net (supervisé)
python train_unet.py

# VAE (supervisé)
python train_vae.py

# Noise2Void (self-supervisé — ne nécessite pas d'images propres)
python train_n2v.py
```

## Évaluation

```bash
# Évaluer le U-Net
python evaluate_models.py

# Évaluer le VAE
python evaluate_vae.py
```

Les notebooks d'analyse comparative (PSNR/SSIM/LPIPS, OOD, hautes fréquences) sont disponibles dans `notebooks_analysis/`.

## Résultats principaux

| Modèle | PSNR σ=25 | PSNR σ=100 (OOD) | Sel & poivre | Flou gaussien |
|--------|-----------|-------------------|--------------|----------------|
| U-Net  | 26,96 dB  | 15,53 dB          | 19,87 dB     | 19,35 dB       |
| VAE    | 27,07 dB  | **21,20 dB**      | 19,54 dB     | **19,84 dB**   |
| N2V    | 22,63 dB  | 17,91 dB          | **21,49 dB** | 19,58 dB       |

## Visualisations

Les graphiques (courbes de loss, exemples d'images débruitées, analyses hautes fréquences, heatmaps OOD) sont automatiquement enregistrés et disponibles.

## Auteurs

Tania Admane, Tea Toscan Du Plantier, Ndoumbé Bayo

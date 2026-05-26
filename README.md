# Deep-learning

La data que j'ai utilisé provient de https://ofsoundof.github.io/lsdir-data/. Je n'ai utilisé que les 4000 premieres images du train pour des raisons de stockages sur onycia (5000 premeires en HD) et la validation. 

la data doit etre du format 

```txt
data 
    train
        clean
        degraded
    val
        clean
        degraded 
```
Si sur mon compte onyxia commencer par telecharger la data en local : 
```bash
#!/bin/bash

# Nettoyer
rm -rf ~/work/data
echo " Création de la structure..."
mkdir -p ~/work/data/train/clean
mkdir -p ~/work/data/val/clean

# Télécharger 5 dossiers pour train (~5000 images HR)
mc cp --recursive s3/taniaadmane/dossier/train/clean/0001000/ ~/work/data/train/clean/0001000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0002000/ ~/work/data/train/clean/0002000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0003000/ ~/work/data/train/clean/0003000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0004000/ ~/work/data/train/clean/0004000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0005000/ ~/work/data/train/clean/0005000/

# Validation (clean seulement)
mc cp --recursive s3/taniaadmane/dossier/val/clean/ ~/work/data/val/clean/

# Vérifier
echo ""
echo "VÉRIFICATION DU TÉLÉCHARGEMENT"
echo "=================================="
echo "=== Train clean ==="
find ~/work/data/train/clean -name "*.png" | wc -l
echo ""
echo "=== Val clean ==="
find ~/work/data/val/clean -name "*.png" | wc -l
echo ""
echo "=== Espace disque ==="
df -h ~/work
echo ""
echo "Téléchargement terminé!"
echo ""
echo " Prochaines étapes:"
echo "  1. Créer dataset bruité: python creer_bruitt.py"
echo "  2. Lancer entraînement: python main.py train"
```
lancer la creation du bruit et Lancer l'entrainement

```bash
python main_unet_vae.py
```

l'analyse est dans analysis.ipynb

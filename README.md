# Deep-learning

La data que j'ai utilisé provient de https://ofsoundof.github.io/lsdir-data/. Je n'ai utilisé que les 4000 premieres images du train pour des raisons de stockages sur onycia (4000 premeires en HD, 4000 premieres bruité x2) et la validation. 

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
# Nettoyer
rm -rf ~/work/data
# Créer structure
mkdir -p ~/work/data/train/{clean,degraded}
mkdir -p ~/work/data/val/{clean,degraded}
# Télécharger 3 dossiers pour train (~3000 images)
mc cp --recursive s3/taniaadmane/dossier/train/clean/0001000/ ~/work/data/train/clean/0001000/
mc cp --recursive s3/taniaadmane/dossier/train/degraded/0001000/ ~/work/data/train/degraded/0001000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0002000/ ~/work/data/train/clean/0002000/
mc cp --recursive s3/taniaadmane/dossier/train/degraded/0002000/ ~/work/data/train/degraded/0002000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0003000/ ~/work/data/train/clean/0003000/
mc cp --recursive s3/taniaadmane/dossier/train/degraded/0003000/ ~/work/data/train/degraded/0003000/
mc cp --recursive s3/taniaadmane/dossier/train/clean/0004000/ ~/work/data/train/clean/0004000/
mc cp --recursive s3/taniaadmane/dossier/train/degraded/0004000/ ~/work/data/train/degraded/0004000/
# Validation 
mc cp --recursive s3/taniaadmane/dossier/val/clean/ ~/work/data/val/clean/
mc cp --recursive s3/taniaadmane/dossier/val/degraded/ ~/work/data/val/degraded/
# Vérifier
echo "=== Train clean ==="
find ~/work/data/train/clean -name "*.png" | wc -l
echo "=== Train degraded ==="
find ~/work/data/train/degraded -name "*.png" | wc -l
echo "=== Val clean ==="
find ~/work/data/val/clean -name "*.png" | wc -l
echo "=== Val degraded ==="
find ~/work/data/val/degraded -name "*.png" | wc -l
echo "=== Espace disque ==="
df -h ~/work
```
pour lancer l'entrainement 

```bash
python main.py train
```

mais les resultats sont cata a corriger
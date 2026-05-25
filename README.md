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

pour lancer l'entrainement 

```bash
python main.py train
```

mais les resultats sont cata a corriger
# Transaction labeling



Classer une transaction carte dans une catégorie (`minor_category`, 12 classes)
à partir du nom du marchand, et en déduire la catégorie large (`major_category`).
Approche : nettoyage du nom (dont le bruit des processeurs de paiement) ->
TF-IDF au niveau caractère -> réduction SVD -> RandomForest. `minor -> major`
étant déterministe, un seul modèle est entraîné et le major est déduit par table.

Résultat : accuracy =0.91, macro-F1 =0.89 (split temporel).

Le notebook transaction_labeling.ipynb indique pas à pas les explorations, la prédiction du modèle et ce que j'aurais fait si j'avais eu  plus de temps. 

## Lancer l'API 

```bash

# 1. Créer et activer un environnement virtuel
python -m venv .venv
#source .venv/bin/activate        # macOS / Linux
 .venv\Scripts\activate         # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'API
uvicorn app:app --port 8000

```

Puis ouvrir **http://127.0.0.1:8000/docs** : interface de test, déplier
`POST /predict`, cliquer *Try it out*, *Execute*.

L'API entraîne le modèle au démarrage à partir de `data/`. et le
sauvegarde dans `model.joblib`. Les démarrages suivants le rechargent (instantané).. Attendre le message
`Application startup complete.` avant d'envoyer une requête.


### Exemple de requête

On ouvre un autre Terminal on active l'environnement virtuel créer ci-dessus et on lance une requête pour intéroger le modèle afin d'obtenir une prédiction de class. 

macOS / Linux :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "BETCLIC", "amount_value": -50}'
```

Windows PowerShell :

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" `
  -Body '{"title": "BETCLIC", "amount_value": -50}'
```

Réponse :

```json
{"minor_category": "betting", "major_category": "entertainment", "confidence": 1.0}
```

## Exploration & analyse

`transaction_labeling.ipynb` contient l'exploration des données, les choix de
modélisation, l'évaluation, et une section "ce que je ferais avec plus de temps".

## Fichiers

```
app.py                       service FastAPI (entraine au demarrage, /predict)
transaction_labeling.ipynb   exploration + entrainement + evaluation
data/transactions.csv        jeu fourni
requirements.txt
```


## Choix et trade-offs

J'ai choisi un modèle RandomForest pour plusieurs raisons : robustesse aux valeurs aberrantes; ses séparations reposent sur des seuils (supérieur/inférieur), donc un montant extrême tombe simplement du même côté d'une coupe sans déformer le modèle, contrairement à un modèle linéaire ou à distance. Il capture aussi des interactions non linéaires entre features et donne de bons résultats sans réglage fin des hyperparamètres, ce qui est précieux sur un temps court. En contrepartie, il est plus lourd qu'un modèle linéaire et moins à l'aise sur des features texte très creuses : c'est pourquoi j'insère une SVD pour réduire les dimensions afin de densifier le TF-IDF pour le passer infine au modèle. 

Pour le texte, j'ai choisi un TF-IDF au niveau caractère plutôt que mot : les noms de marchands sont des identifiants collés ou tronqués (CARREFOURMARKET, UBR*), pas des phrases, et les n-grams de caractères y sont plus robustes. La limite, c'est qu'aucune représentation du nom seul ne permet de classer un marchand réellement inconnu sans information exploitable, ce que je traiterais par enrichissement externe (voir pistes futures dans le notebook d'exploration).

Je ne prédis que minor_category et je déduis major_category par table, puisque la correspondance est déterministe : un seul modèle, et aucune incohérence possible entre les deux niveaux. J'évalue en macro-F1 plutôt qu'en accuracy, car les classes sont déséquilibrées et l'accuracy serait flattée par les classes fréquentes. Enfin, je teste en split temporel (passé vers futur) pour rester proche de la production ; un split par marchand mesurerait en plus la généralisation aux marchands inconnus, que je garde en piste future.
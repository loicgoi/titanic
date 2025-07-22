# Titanic – Machine Learning from Disaster

**Objectif**  
Prédire la survie des passagers du Titanic (0 ou 1) à partir des données Kaggle.

---

## 📁 Arborescence
.
├── data/
│  ├── train.csv          # jeu d’entraînement
│   └── test.csv           # jeu de test
├── titanic.ipynb         # notebook principal
├── submission.csv         # fichier généré pour Kaggle
└── requirements.txt       # fichier pour les dépendances

---

## ⚙️ Installation

pip install -r requirements.txt   # pandas, scikit-learn, seaborn, matplotlib
🚀 Usage rapide
Python

from pipeline import pipeline
import joblib

x, y, x_test, ids, _, _ = pipeline(df_train_raw, df_test_raw)

model = joblib.load("titanic_rf.pkl")
y_pred = model.predict(x_test)

🧪 Pipeline
Chargement auto des CSV
Feature engineering (titre, taille famille, médianes)
Imputation intelligente
Encodage & nettoyage
Modèle RandomForest optimisé (RandomizedSearchCV)
Génération submission.csv

📈 Performance (80 % échantillon)
Modèle	Accuracy CV	Log-loss
RandomForest (tuned)	0.835	0.37

📝 Générer la soumission
bash

python -c "from pipeline import save_csv_submission; save_csv_submission(ids, y_pred)"


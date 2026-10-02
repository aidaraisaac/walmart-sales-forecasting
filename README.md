# 🛒 Walmart Retail Analytics — Prévision des ventes & Recommandation de réapprovisionnement

Système d'aide à la décision pour une enseigne de grande distribution : prévoir la demande hebdomadaire par magasin/rayon et en déduire des recommandations de réapprovisionnement et de mise en avant produit.

## 📌 Contexte & objectif

Une PME de distribution doit arbitrer en permanence entre deux risques : la **rupture de stock** (ventes perdues, client insatisfait) et le **surstock** (capital immobilisé, invendus). Ce projet construit un pipeline complet — de la donnée brute à la recommandation actionnable — pour objectiver cet arbitrage à partir de l'historique de ventes.

Deux briques complémentaires :
1. **Prévision** — anticiper les ventes hebdomadaires par magasin et par rayon.
2. **Recommandation** — traduire ces prévisions en priorités de réapprovisionnement et de mise en avant.

## 📊 Données

Dataset : [Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (Kaggle).

| Fichier | Contenu |
|---|---|
| `train.csv` | Ventes hebdomadaires par magasin/rayon (`Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`) |
| `features.csv` | Variables exogènes par magasin/semaine (température, prix du carburant, promotions `MarkDown1-5`, CPI, taux de chômage) |
| `stores.csv` | Métadonnées des 45 magasins (`Type`, `Size`) |

Période couverte : février 2010 – novembre 2012, 45 magasins, ~81 rayons.

> ⚠️ Le dataset ne contient **aucune donnée de stock/inventaire**. La brique recommandation repose donc sur des hypothèses explicites (politique de stock de sécurité, délai de réapprovisionnement simulé) — détaillées dans `notebooks/04_recommendation_engine.ipynb` et assumées comme telles plutôt que présentées comme des données réelles.

## 🏗️ Structure du projet

```
walmart-sales-forecasting-recommender/
├── data/
│   ├── raw/                  # train.csv, features.csv, stores.csv (non versionnés)
│   └── processed/            # dataset fusionné et nettoyé (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb                     # exploration & visualisation
│   ├── 02_feature_engineering.ipynb     # lags, moyennes mobiles, variables calendaires
│   ├── 03_forecasting_models.ipynb      # baseline → SARIMA/Prophet → XGBoost
│   └── 04_recommendation_engine.ipynb   # score de priorité de réapprovisionnement
├── src/
│   ├── data_loader.py         # chargement & fusion des 3 fichiers sources
│   ├── features.py            # feature engineering réutilisable
│   ├── forecasting.py         # entraînement & évaluation des modèles
│   └── recommender.py         # logique de scoring réapprovisionnement
├── dashboard/
│   └── app.py                 # tableau de bord Streamlit
├── reports/
│   └── figures/                # graphiques exportés
├── requirements.txt
└── README.md
```

Les notebooks servent à l'exploration et à la narration ; la logique réutilisable est extraite dans `src/` et importée depuis les notebooks — pratique standard pour garder un code testable plutôt que tout empiler dans des cellules.

## 🔬 Méthodologie

1. **EDA** — tendance globale, saisonnalité, effet jours fériés, écarts entre types de magasins (A/B/C).
2. **Feature engineering** — lags (S-1, S-52), moyennes mobiles, indicateurs calendaires (fêtes US : Super Bowl, Thanksgiving, Noël), encodage du type de magasin.
3. **Modélisation de la prévision** — baseline naïve saisonnière, puis comparaison SARIMA / Prophet / XGBoost. Évaluation avec la métrique officielle de la compétition, le **WMAE** (Weighted Mean Absolute Error, poids ×5 sur les semaines de fêtes) plutôt qu'un simple MAE, pour rester fidèle à l'enjeu métier réel (les erreurs pendant les pics de vente coûtent plus cher).
4. **Moteur de recommandation** — score de priorité = f(croissance de la demande prévue, niveau de stock simulé, marge produit), avec hypothèses de simulation documentées.
5. **Dashboard** — visualisation interactive des prévisions et des alertes de réapprovisionnement par magasin/rayon.

## ⚙️ Stack technique

Python · pandas · numpy · matplotlib/seaborn · scikit-learn · statsmodels / Prophet · XGBoost · Streamlit · Jupyter (VS Code)

## 🚀 Installation & utilisation

```bash
git clone <url-du-repo>
cd walmart-sales-forecasting-recommender
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Dans VS Code : installer l'extension **Jupyter**, ouvrir un `.ipynb` dans `notebooks/`, puis sélectionner l'interpréteur `venv` comme kernel (palette de commandes → *Python: Select Interpreter*). Exécuter les notebooks dans l'ordre numéroté.

Pour le dashboard :
```bash
streamlit run dashboard/app.py
```

## 📈 Résultats clés

_À compléter une fois les modèles entraînés — WMAE du modèle retenu vs baseline, gain estimé en réduction de rupture de stock, etc._

## 🎯 Compétences démontrées

- Nettoyage et fusion de sources de données hétérogènes
- Feature engineering pour séries temporelles
- Modélisation et comparaison rigoureuse de modèles de prévision (métrique métier)
- Conception d'un système de recommandation à partir de règles/scores
- Visualisation et restitution via dashboard interactif
- Code modulaire et reproductible (notebooks + package `src/`)

## 🔭 Limites & pistes d'amélioration

- Pas de données de stock réelles → à remplacer par de vraies données d'inventaire si le projet est adapté à un cas d'usage réel.
- Granularité limitée aux variables du dataset (pas de données concurrentielles, météo fine, etc.).
- Pistes : hiérarchisation des prévisions (reconciliation top-down/bottom-up), passage à un modèle deep learning (N-BEATS, Temporal Fusion Transformer) si le temps le permet.

## 👤 Auteur

BANDE Aïdara Isaac — Étudiant en Génie informatique, École Polytechnique de Ouagadougou

# ML-volatility-trading-using-long-and-short-straddle

Ce projet a été réalisé dans le cadre du cours **Méthodes d’apprentissage appliquées aux données financières (MATH 60610)** à HEC Montréal.

L’objectif est de prédire les écarts entre la volatilité implicite et la volatilité réalisée du S&P 500 sur des horizons de **10, 30 et 60 jours**, puis d’utiliser ces prévisions pour générer des signaux de trading sur des straddles ATM.

Le projet combine plusieurs techniques d’apprentissage automatique, notamment **CatBoost, Ridge, Random Forest, SVR, régression logistique et K-means**. Les modèles utilisent des données provenant du SPX, des options, de la structure de volatilité et de plusieurs variables macroéconomiques.

Les modèles sont entraînés et validés sur la période **2010–2018**, puis la stratégie est testée hors échantillon entre **2018 et 2023**.

Le dépôt contient le code de préparation des données, d’entraînement des modèles, de classification des signaux et de backtest.

## Équipe

Projet réalisé par :
- Alpha Amadou Diallo
- Anthony Touville
- Ludwig Casaubon
- Pierre Louis Nouvellon

## Données

Les données proviennent principalement de **OptionMetrics/WRDS, Yahoo Finance et FRED**. Certaines données propriétaires ne sont pas incluses dans le dépôt.

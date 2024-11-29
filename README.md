# Projet de Prédiction de Tendance avec TensorFlow et Apprentissage par Renforcement

Ce projet implémente un modèle de prédiction de tendance basé sur les articles et l'analyse du sentiment à l'aide de TensorFlow et de l'apprentissage par renforcement. Le but est de prédire la tendance d'un actif financier (par exemple, Dogecoin) en se basant sur les sentiments extraits de contenus textuels (articles, tweets, etc.).

## Table des matières

- [Contexte du projet](#contexte-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Explication du code](#explication-du-code)

## Contexte du projet

Le modèle utilise les scores de sentiment d'articles pour prédire la tendance d'un actif financier. Il utilise l'apprentissage par renforcement pour améliorer ses prédictions au fil du temps, en ajustant sa prise de décision et en apprenant des erreurs passées.

## Prérequis

Avant de pouvoir exécuter le projet, assurez-vous que vous disposez des éléments suivants :
- **Node.js** (version 16 ou supérieure)
- **npm** (gestionnaire de paquets pour Node.js)
- Un **compte Hugging Face** et une **clé API** pour l'analyse de sentiment

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/username/projet-prediction-tendance.git
   cd projet-prediction-tendance
Installez les dépendances du projet :

bash
npm install
Créez un fichier .env à la racine de votre projet et ajoutez-y votre clé API Hugging Face :

env
API_KEY=your_huggingface_api_key

## Utilisation

Démarrer la prédiction
Chargez les articles à partir du fichier dogecoin_articles_perte.json dans le dossier racine du projet.
Lancez le processus de prédiction en exécutant la commande suivante :
bash
node index.js
Cela démarrera l'entraînement et les prédictions du modèle, et les résultats seront logués dans le fichier predictions_log.txt.

Sauvegarder et charger le modèle
Le modèle est sauvegardé localement après chaque entraînement dans le dossier model/. Vous pouvez le charger à tout moment pour continuer les prédictions ou l'entraînement.

javascript
await model.loadModel();  // Charge le modèle sauvegardé
await model.saveModel();  // Sauvegarde le modèle après l'entraînement

##Explication du code

Le modèle suit un flux de travail basé sur l'apprentissage par renforcement avec la mémorisation des expériences pour ajuster ses actions en fonction de la récompense qu'il reçoit :

Analyse de sentiment : Les sentiments des articles sont extraits à l'aide de l'API Hugging Face pour évaluer si un article est positif, négatif ou neutre.
Prédiction de la tendance : Le modèle prend en compte le sentiment global pour prédire la tendance d'un actif (par exemple, si la tendance est haussière ou baissière).
Apprentissage par renforcement : Les erreurs sont analysées et le modèle ajuste ses prédictions via une mémoire d'expérience pour éviter de refaire les mêmes erreurs.
Le modèle est conçu pour apprendre des erreurs passées, et une récompense est attribuée en fonction de la précision des prédictions.


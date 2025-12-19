# SMS Spam Detection: Synthetic Data Augmentation with LLMs

Ce projet explore l'utilisation des modèles de langage (LLMs) pour pallier le manque de données dans les tâches de classification de texte. L'objectif est de comparer les performances d'un classifieur entraîné sur un jeu de données réel restreint par rapport à un modèle enrichi par des données synthétiques.

## 📌 Objectifs du Projet
1.  **Baseline :** Mettre en place un pipeline de classification "lightweight" (SVM, Random Forest) sur le dataset *SMS Spam Collection*.
2.  **Contrainte :** Sub-échantillonner le dataset pour simuler une pénurie de données.
3.  **Augmentation :** Utiliser un LLM (via prompting) pour générer des SMS synthétiques réalistes (Spam et Ham).
4.  **Évaluation :** Analyser l'impact de l'augmentation sur la précision, particulièrement pour la classe minoritaire, en utilisant des métriques adaptées (F1-Score, MCC).

## 🛠️ Stack Technique
* **Langage :** Python 3.x
* **NLP & ML :** `scikit-learn`, `pandas`, `nltk` (ou `spaCy`)
* **Embeddings :** TF-IDF / N-Grams (priorité à la simplicité pour observer le gain de l'augmentation)
* **Génération :** OpenAI GPT, Google Gemini ou Llama 3 (local)

## 📂 Structure du Répertoire
* `data/` : Contient le dataset original et les échantillons réduits.
* `notebooks/` : Expérimentations et visualisation des résultats.
* `src/` : Scripts de prétraitement, d'entraînement et de génération de prompts.
* `synthetic_data/` : SMS générés par le LLM.

## 🚀 Méthodologie

### 1. Prétraitement & Sub-échantillonnage
* Nettoyage des SMS (minuscules, suppression de la ponctuation, stop-words).
* Réduction drastique de la taille du set d'entraînement (ex: 50-100 exemples) pour créer un besoin de données.
* Séparation stricte d'un set de validation (données réelles uniquement).

### 2. Stratégies de Prompting
Deux approches sont testées pour la génération synthétique :
* **Knowledge-based :** Demander au LLM de générer des spams basés sur sa propre connaissance interne.
* **Few-shot prompting :** Fournir quelques exemples réels du dataset au LLM pour qu'il imite le style et le vocabulaire.

###

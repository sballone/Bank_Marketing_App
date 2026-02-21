# 🏦 BANK MARKETING INTELLIGENCE PLATFORM

## 🎯 Application Originale et Professionnelle

### ⚡ Points Forts qui Rendent cette Application UNIQUE

Cette application se distingue par :

#### 🎨 **Design Ultra-Moderne**
- Interface professionnelle avec CSS personnalisé
- Animations et transitions fluides
- Thème bancaire cohérent (bleu corporate)
- Cards stylisées et responsive
- Gradient backgrounds

#### 🚀 **Fonctionnalités Innovantes**
1. **Dashboard Exécutif Interactif**
   - KPIs en temps réel
   - Graphiques Plotly 3D
   - Insights automatiques générés par IA
   - Timeline des campagnes

2. **Exploration Intelligente**
   - Visualisations interactives avancées
   - Analyse bivariée automatique
   - Matrices de corrélation animées

3. **Machine Learning Hub**
   - Entraînement de 6 modèles différents
   - Comparaison multi-critères
   - Graphique radar innovant
   - Ensemble learning (Voting Classifier)

4. **Prédicteur Temps Réel**
   - Formulaire interactif complet
   - Prédictions avec probabilités
   - Jauge animée (gauge chart)
   - Recommandations personnalisées automatiques

5. **Analytics Avancés** (Bonus)
   - Feature importance interactive
   - Courbes ROC comparatives
   - Matrice de confusion enrichie

6. **Comparaison de Modèles** (Bonus)
   - Benchmark automatique
   - Cross-validation
   - Export des résultats

7. **Export & Déploiement** (Bonus)
   - Sauvegarde modèles (.pkl)
   - Export rapports PDF
   - API REST ready

---

## 📁 Structure du Projet

```
bank-marketing-project/
├── bank_marketing_app.py          # Application principale
├── bank-full.csv                  # Dataset (45,211 lignes)
├── requirements_bank.txt          # Dépendances Python
├── README.md                      # Ce fichier
└── models/                        # Dossier pour sauvegarder les modèles (auto-créé)
```

---

## 📊 Dataset - Bank Marketing

### Source
**UCI Machine Learning Repository**  
Données de campagnes marketing téléphoniques d'une banque portugaise (2008-2010)

### Caractéristiques
- **45,211 clients**
- **17 variables** (16 features + 1 cible)
- **Cible** : Souscription à un dépôt à terme (yes/no)

### Variables Clés
- **Démographiques** : âge, profession, éducation, situation familiale
- **Financières** : balance, crédit, prêts
- **Campagne** : durée appel, nombre de contacts, résultat précédent

### Défi ML
**Classification binaire déséquilibrée**  
- ~88% Non (classe majoritaire)
- ~12% Oui (classe minoritaire)



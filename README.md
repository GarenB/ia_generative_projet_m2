# Analyste Financier Llama

### Objectif:

L'objectif de ce projet est de concevoir un outil d'aide à la décision financière capable de recommander l'achat d'une action d'une entreprise donnée, ou de l'éviter, en s'appuyant sur des données financières extraites entre 2024 et 2025.

Pour cela, nous avons développé une architecture multi-agents où chaque agent possède un rôle prédéfini et des règles strictes.

#### 1. l'Analyste de marché optimiste

Son rôle est de présenter l'entreprise sous son meilleur jour pour la période 2024-2025. Il explore les actualités financières provenant de sources de confiance (CNBC, Reuters, Bloomberg...) ainsi que les rapports annuels officiels. Il parvient à une solution de la forme suivante:

- La performance financière
- Les parts de marché
- La trésorerie

#### 2. l'Auditeur de risques

Son rôle est d'identifier les menaces et les indicateurs de ralentissement pour la période 2024-2025 en utilisant les mêmes sources de confiance. Il parvient à une solution de la forme suivante:

- La santé financière
- Les operations (restructurations, fermetures de sites...)
- Les risques de marché

#### 3. L'Arbitre d'investissement

Son rôle est de trancher entre les arguments de l'analyste optimiste et de l'auditeur de risques via un processus de réflexion critique.
Son analyse doit se porter sur les caractéristiques suivantes:

- Un investissement à moyen/long terme (3 à 5 ans)
- Une croissance durable
- Une solidité du bilan

### Les Techniques de Raisonnement utilisées:

#### 1. Chain of Thought (CoT)

Nous utilisons le CoT car une décision d’investissement est un problème complexe, et cette approche permet d’obtenir un raisonnement structuré et plus fiable qu’une réponse directe. D’abord, l'arbitre analyse les données financières, puis il identifie les variables importantes, et enfin il construit une décision cohérente.

#### 2. ReAct (Reason + Act)

Nous utilisons ReAct pour éviter les hallucinations et l’utilisation d’informations obsolètes, car une analyse financière nécessite des données réelles et récentes. Lorsqu'un agent a besoin d’informations financières, il peut agir en appelant un outil externe (TavilySearch).

#### 3. Self-Correction (reflexion)

Nous utilisons la Self-Correction pour améliorer la qualité de la décision finale. Avant de donner son verdict, l'arbitre doit critiquer son propre raisonnement initial pour détecter d’éventuels biais ou incohérences. Il ajuste ensuite son verdict et ses scores de confiance en fonction de cette réflexion.

### Commande pour lancer le code:

streamlit run app.py

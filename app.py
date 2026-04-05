import base64
import json
import ast
import os
import matplotlib.pyplot as plt
import io
from langchain_tavily import TavilySearch
import numpy as np
import streamlit as st
import operator
import yfinance as yf
import requests
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

@tool
def analyze_financial_chart(ticker: str):
    """
    Génère un graphique boursier de 2024 à 2025 pour un ticker donné,
    puis l'analyse grâce à un modèle HuggingFace spécialisé en lecture de graphes.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start="2024-01-01", end="2025-12-31")

        if hist.empty:
            return "Aucune donnée disponible pour ce ticker."

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist.index, hist["Close"], color="blue", linewidth=1.5)
        ax.set_title(f"Cours de {ticker} — 2024-2025")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix ($)")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_data = buf.read()
        buf.seek(0)
        st.image(buf, caption=f"Cours de {ticker} 2024-2025")
        plt.close()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        API_URL = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        # Ajouter un modèle HuggingFace spécialisé en lecture de graphes financiers (nécessite un compte payant)
        response = requests.post(API_URL, headers=headers, json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyse ce graphique boursier. Décris la tendance générale (haussière, baissière, volatile), les points clés et ce que cela signifie pour un investisseur."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "model": ""
        })

        if response.status_code == 200:
            analysis = response.json()["choices"][0]["message"]["content"]
            return f"Analyse du graphique {ticker} : {analysis}"
        else:
            return f"Modèle HuggingFace indisponible (status {response.status_code}). Ignore l'analyse visuelle."

    except Exception as e:
        return f"Erreur : {e}"

@tool
def fetch_financial_metrics(ticker: str):
    """
    Récupère les données financières réelles (Prix, PE Ratio, Dette...) pour un ticker boursier donné.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        metrics = {
            "nom": info.get("longName"),
            "cours": info.get("currentPrice"),
            "PE_Ratio": info.get("trailingPE"),
            "Debt_to_Equity": info.get("debtToEquity"),
            "Marge_Net": info.get("profitMargins"),
            "Cash_Flow": info.get("freeCashflow"),
            "Revenue_Growth": info.get("revenueGrowth")
        }
        return str(metrics)
    except Exception as e:
        return f"Erreur yfinance: {e}"

def get_final_decision(debat, nb_samples=2):
    samples = []
    for _ in range(nb_samples):
        res = agent_moderateur.graph.invoke({"messages": [HumanMessage(content=debat)]})
        raw_content = res['messages'][-1].content
        
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(clean_content)
            samples.append(parsed)
        except:
            continue 

    if not samples:
        return {
            "verdict": "INDÉTERMINÉ", "achat": 0, "eviter": 0, 
            "explication": "Erreur de lecture du modèle.", "declic": "N/A"
        }

    return {
        "verdict": samples[-1]["verdict"],
        "achat": np.mean([s["score_achat"] for s in samples]),
        "eviter": np.mean([s["score_eviter"] for s in samples]),
        "explication": samples[-1].get("summary", ""),
        "declic": samples[-1].get("arguments", "")
    }

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools) if tools else model

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(getattr(result, 'tool_calls', [])) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] in self.tools:
                result = self.tools[t['name']].invoke(t['args'])
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

model = ChatOpenAI(model="gpt-4o", temperature=0.8)

prompt_bull = """
Rôle : Analyste de Marché Optimiste (Très Optimiste).
Mission : Présenter {entreprise} sous son meilleur jour possible pour la période 2024-2025.

OUTILS DISPONIBLES :
- TavilySearch : pour trouver des actualités financières récentes (CNBC, Reuters, Bloomberg...).
- fetch_financial_metrics : pour récupérer les chiffres financiers réels.
  → Trouve d'abord le ticker boursier officiel de l'entreprise
    (ex: NVIDIA → NVDA, Apple → AAPL, Tesla → TSLA, Microsoft → MSFT).
  → Appelle cet outil EN PREMIER avec le bon ticker, puis TavilySearch pour le contexte.
- analyze_financial_chart : pour analyser visuellement le graphique boursier sur la période 2024-2025.
  → Utilise le même ticker que fetch_financial_metrics (ex: NVDA, AAPL...).
  → Appelle cet outil EN DERNIER, après fetch_financial_metrics et TavilySearch.

RÈGLES :
1. **RÉSULTATS FACTUELS** : Cherche les chiffres de Chiffre d'Affaires (CA) et de Bénéfice Net dans les news financières (CNBC, Reuters, Bloomberg, Yahoo Finance...) si les rapports officiels sont trop denses.
2. **PAS DE PRÉVISIONS** : Ne cite que des chiffres de trimestres déjà clôturés.
3. **CONTEXTE** : Si l'entreprise bat des records, mentionne-le avec le chiffre précis.
4. **SOURCES** : Cite le média et la date pour chaque donnée. INTERDICTION d'inventer des chiffres.
5. **FILTRE POSITIF** : Interdiction formelle de mentionner des faits négatifs.

STRUCTURE :
- Performance financière (CA, Marges, Profits).
- Leadership (Parts de marché, nouveaux contrats signés).
- Trésorerie (Cash disponible).
"""

prompt_bear = """
Rôle : Auditeur de Risques (Zéro Spéculation). 
Mission : Identifier les menaces financières factuelles et les indicateurs de ralentissement de {entreprise} pour la période 2024-2025.

OUTILS DISPONIBLES :
- TavilySearch : pour trouver des actualités financières récentes (CNBC, Reuters, Bloomberg...).
- fetch_financial_metrics : pour récupérer les chiffres financiers réels.
  → Trouve d'abord le ticker boursier officiel de l'entreprise
    (ex: NVIDIA → NVDA, Apple → AAPL, Tesla → TSLA, Microsoft → MSFT).
  → Appelle cet outil EN PREMIER avec le bon ticker, puis TavilySearch pour le contexte.
- analyze_financial_chart : pour analyser visuellement le graphique boursier sur la période 2024-2025.
  → Utilise le même ticker que fetch_financial_metrics (ex: NVDA, AAPL...).
  → Appelle cet outil EN DERNIER, après fetch_financial_metrics et TavilySearch.

RÈGLES DE FER : 
1. **DANGER RÉEL** : Cite les dettes, pertes nettes, licenciements, MAIS AUSSI toute baisse de chiffre d'affaires ou perte de parts de marché publiée.
2. **PAS DE PRÉVISIONS** : Uniquement des faits actés (trimestres clos, annonces officielles).
3. **SOUPLESSE DE RECHERCHE** : Si aucun risque de faillite n'existe, cherche les litiges juridiques, les amendes ou la baisse des marges opérationnelles.
4. **RÉSULTATS FACTUELS** : Cherche les chiffres dans les news financières (CNBC, Reuters, Bloomberg, Yahoo Finance...) si les rapports officiels sont trop denses.
5. **SOURCES** : Cite le média et la date pour chaque donnée. INTERDICTION d'inventer des chiffres.

STRUCTURE :
- **Santé Financière** : Endettement, pertes nettes ou baisse de revenus (chiffres précis).
- **Opérations** : Restructurations, fermetures de sites ou suppressions de postes.
- **Risques de Marché** : Chute du cours de l'action, litiges en cours ou pression concurrentielle documentée.
"""

prompt_moderateur = """
Rôle : Arbitre d'Investissement.
Mission : Trancher entre les arguments BULL et BEAR via un processus de réflexion critique.
Contexte : Ton analyse doit se porter sur un investissement à moyen/long terme (3 à 5 ans). Ne privilégie pas les gains spéculatifs de court terme, mais la croissance durable et la solidité du bilan.

### MÉTHODE DE RAISONNEMENT (Chain of Thought & Self-Correction) :
Tu dois décomposer ton problème étape par étape avant de conclure :

1. **ANALYSE LES DONNÉES** : Extrais les chiffres clés (CA, Profits, Dettes, Cash) du BULL et du BEAR.
2. **IDENTIFIE LES VARIABLES** : Compare les forces financières face aux faiblesses réelles.
3. **CALCULE LA SOLUTION** : Détermine une première décision (Achat ou Éviter).

4. **AUTO-CRITIQUE (Self-Correction)** : 
   Analyse ta propre décision. Vérifie s'il y a des incohérences logiques, des biais (trop optimiste/pessimiste) ou un manque d'information cruciale.
5. **CORRIGE TA CONCLUSION** : 
   Ajuste ton verdict final et tes scores si ta critique a révélé une faille.

### RÈGLES D'ARBITRAGE :
1. **SOLVABILITÉ** : Si le Cash > Dette, l'entreprise est en sécurité immédiate.
2. **RENTABILITÉ** : Si le Profit est positif et en croissance, c'est un signal BULL.
3. **ALERTE ROUGE** : Si Dette > Cash ET Pertes nettes cumulées : Signal "À ÉVITER" immédiat (risque de faillite).
4. **EXCEPTION DE CROISSANCE** : Une dette élevée est acceptable UNIQUEMENT si le Chiffre d'Affaires croît plus vite que la dette.
5. **OPPORTUNITÉ DE MARCHÉ** : Si le prix de l'action baisse MAIS que les fondamentaux (Chiffre d'Affaires, bénéfices, cash-flow) sont solides ou en croissance, c'est un signal BULL.
6. **SURÉVALUATION (BULLE)** : Si le prix de l'action monte mais que les bénéfices sont en déclin, c'est un signal "À ÉVITER".
7. **RÈGLE ABSOLUE** : Ne JAMAIS inventer de chiffres. Si une donnée est absente, ignorer la règle concernée.

FORMAT DE RÉPONSE STRICT (JSON UNIQUEMENT) :
{{
  "verdict": "ACHAT IMMÉDIAT" ou "À ÉVITER",
  "score_achat": int (0-100),
  "score_eviter": int (0-100),
  "summary": "\\n 1. ANALYSE : [Texte]\\n 2. DIAGNOSTIC (Forces/Risques) : [Texte]\\n 3. AVIS INITIAL : [Texte]\\n 4. CRITIQUE & CORRECTION : [Texte de ta réflexion interne]",
  "arguments": "Le chiffre clé final qui prouve ta solution après correction."
}}
"""

search_tool = TavilySearch(max_results=2)
finance_tool = fetch_financial_metrics
vision_tool = analyze_financial_chart

search_tools = [search_tool, finance_tool, vision_tool]

agent_optimiste = Agent(model, search_tools, system=prompt_bull)
agent_pessimiste = Agent(model, search_tools, system=prompt_bear)
agent_moderateur = Agent(model, [], system=prompt_moderateur)

st.set_page_config(page_title="Analyste Financier GPT-4o")
st.title("Analyste Financier GPT-4o")
entreprise = st.text_input("Entreprise :", placeholder="ex: NVIDIA...")

if st.button("Lancer l'investigation") and entreprise:
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("L'optimiste cherche..."):
            res_opt = agent_optimiste.graph.invoke({"messages": [HumanMessage(content=f"Analyse {entreprise}")]})
            st.success(res_opt['messages'][-1].content)
            
    with col2:
        with st.spinner("Le sceptique cherche..."):
            res_pess = agent_pessimiste.graph.invoke({"messages": [HumanMessage(content=f"Risques {entreprise}")]})
            st.error(res_pess['messages'][-1].content)

    st.divider() 
    
    exp_opt, exp_pess = st.tabs(["🔗 Sources Optimistes", "🔗 Sources Pessimistes"])
    
    with exp_opt:
        for m in res_opt['messages']:
            if isinstance(m, ToolMessage):
                if m.name == "fetch_financial_metrics":
                    st.markdown("**Données Yahoo Finance**")
                    st.code(m.content, language=None)
                elif m.name == "analyze_financial_chart":
                    st.markdown("**Analyse HuggingFace**")
                    st.write(m.content)
                else:
                    try:
                        data = ast.literal_eval(m.content)
                        for s in data.get('results', []):
                            st.markdown(f"- **{s['title']}** ([Lien]({s['url']}))")
                    except:
                        st.write(m.content)

    with exp_pess:
        for m in res_pess['messages']:
            if isinstance(m, ToolMessage):
                if m.name == "fetch_financial_metrics":
                    st.markdown("**Données Yahoo Finance**")
                    st.code(m.content, language=None)
                elif m.name == "analyze_financial_chart":
                    st.markdown("**Analyse HuggingFace**")
                    st.write(m.content)
                else:
                    try:
                        data = ast.literal_eval(m.content)
                        for s in data.get('results', []):
                            st.markdown(f"- **{s['title']}** ([Lien]({s['url']}))")
                    except:
                        st.write(m.content)
    st.divider()
    with st.spinner("Décision de l'arbitre en cours..."):
        debat_txt = f"BULL: {res_opt['messages'][-1].content}\nBEAR: {res_pess['messages'][-1].content}"
        final = get_final_decision(debat_txt)
        
        score_achat = final['achat']
        score_eviter = final['eviter']
        verdict = final['verdict']

        if "ACHAT" in verdict.upper():
            st.success(f"### ✅ {verdict}")
        else:
            st.error(f"### ❌ {verdict}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Score ACHAT", f"{score_achat:.1f}%")
            st.progress(score_achat / 100)
            
        with col_b:
            st.metric("Score ÉVITER", f"{score_eviter:.1f}%")
            st.progress(score_eviter / 100)

        st.divider()
        st.info(f"**RÉSUMÉ :** {final['explication']}")
        st.warning(f"**ARGUMENT CLÉ :** {final['declic']}")
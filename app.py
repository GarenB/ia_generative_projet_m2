import json

from langchain_tavily import TavilySearch
import numpy as np
import streamlit as st
import operator
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

def get_final_decision(debat, nb_samples=3):
    scores_achat = []
    scores_eviter = []
    explications = []
    
    for _ in range(nb_samples):
        # On utilise ton agent modérateur compilé
        res = agent_moderateur.graph.invoke({"messages": [HumanMessage(content=debat)]})
        raw_content = res['messages'][-1].content
        
        try:
            parsed = json.loads(raw_content)
            scores_achat.append(parsed["score_achat"])
            scores_eviter.append(parsed["score_eviter"])
            explications.append(parsed)
        except:
            continue # On ignore les erreurs de formatage rares

    # On calcule la moyenne des scores pour la stabilité
    final_achat = np.mean(scores_achat)
    final_eviter = np.mean(scores_eviter)
    
    # On prend le dernier verdict pour le texte
    return {
        "verdict": explications[-1]["verdict"],
        "achat": final_achat,
        "eviter": final_eviter,
        "explication": explications[-1]["summary"],
        "declic": explications[-1]["arguments"]
    }

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        # Initialise l'agent avec un modèle de langage, des outils, et une configuration système.
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        # Vérifie si une action est nécessaire après la dernière réponse du modèle.
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        # Appelle le modèle de langage (OpenAI) pour obtenir une réponse en fonction des messages précédents.
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        # Exécute les actions spécifiées par le modèle de langage, en utilisant les outils disponibles.
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# 2. Initialisation du modèle (remplacez par votre clé API)
# Vous pouvez utiliser GPT-4o, Claude, ou même un modèle local via Ollama
model = ChatOpenAI(model="gpt-4o", temperature=0.8)

# 3. Création du Prompt (Le "cerveau" de l'agent)
# On force le raisonnement Chain of Thought (CoT)
prompt_bull = """
Tu es un analyste financier extrêmement optimiste, spécialisé dans la croissance.

TA MISSION : Cherche des news et des rapports financiers récents et explique pourquoi il faut ABSOLUMENT investir.

RÈGLES STRICTES :
1. NE RIEN INVENTER : Si tu n'as pas de chiffre précis via tes outils, ne fais pas de supposition.
2. CITATION DE CHIFFRES : Cite un maximum de chiffres (Chiffre d'affaires, marge, croissance %, part de marché).
3. SOURCES : Indique la source ou la date de chaque donnée citée.
4. PÉRIODE : Uniquement 2024, 2025 et 2026.

Tu dois impérativement suivre cette structure de raisonnement :
1. **Analyse du secteur** : Pourquoi ce marché est-il porteur ?
2. **Points forts de l'entreprise** : Innovation, leadership ou solidité.
3. **Potentiel futur** : Pourquoi l'action va-t-elle monter ?

Sois convaincant et utilise un ton professionnel mais enthousiaste.
"""

prompt_bear = """
Tu es un analyste financier très prudent et sceptique. 

TA MISSION : Cherche des news et des rapports financiers récents et explique pourquoi il ne faut ABSOLUMENT PAS investir ou quels sont les RISQUES majeurs.

RÈGLES STRICTES :
1. NE RIEN INVENTER : Si tu n'as pas de chiffre précis via tes outils, ne fais pas de supposition.
2. CITATION DE CHIFFRES : Cite un maximum de chiffres (Chiffre d'affaires, marge, croissance %, part de marché).
3. SOURCES : Indique la source ou la date de chaque donnée citée.
4. PÉRIODE : Uniquement 2024, 2025 et 2026.

Structure : 
1. Menaces du secteur, 
2. Faiblesses internes, 
3. Risque de perte de valeur.

Sois convaincant et utilise un ton professionnel mais enthousiaste.
"""

prompt_moderateur = """
Tu es un trader de hedge fund agressif. Ton but est le profit à long terme (5 a 10 ans apres l'achat), pas la prudence excessive.
Tu sais que TOUT investissement comporte des risques. 

TA MISSION : Comparer les deux analyses et trancher.

RÈGLES DE DÉCISION :
1. Si le potentiel de gain (Bull) est massivement supérieur aux risques (Bear), choisis ACHAT même si le Bear a des arguments.
2. Ne sois pas une "poule mouillée". Tu es payé pour prendre des decision difficiles. Un risque de 20% n'annule pas une opportunité de 200%.
3. Tranche de manière BRUTALE, PRECISE et BINAIRE.

MÉTHODE DE CALCUL DU SCORE DE CERTITUDE (0-100%) :
1. QUALITÉ DES PREUVES (0-40 pts) : Les chiffres 2024-2026 cités par le camp gagnant sont-ils précis et sourcés ?
2. FORCE DU SIGNAL (0-40 pts) : 
   - Si ACHAT : L'opportunité de profit est-elle massive ? 
   - Si À ÉVITER : Le risque de perte ou la chute des ventes est-il indiscutable ?
3. RÉCENCE (0-20 pts) : Les news majeures datent-elles de moins de 3 mois ?

TA MISSION :
1. Analyse les arguments du Bull et du Bear.
2. Identifie quel camp apporte les preuves chiffrées les plus solides et les plus récentes (2024-2026).
3. Calcule un SCORE DE CERTITUDE au camp "ACHAT" sur les critères ci-dessus.
4. Calcule un SCORE DE CERTITUDE au camp "À ÉVITER" sur les critères ci-dessus.
5. TRANCHE : Tu dois choisir un camp. Pas de "ça dépend".

STRUCTURE DE TA RÉPONSE :
Réponds UNIQUEMENT par un objet JSON respectant ce format :
{
  "verdict": "ACHAT IMMÉDIAT" ou "À ÉVITER",
  "score_achat": SCORE DE CERTITUDE pour le camp ACHAT entre 0 et 100,
  "score_eviter": SCORE DE CERTITUDE pour le camp À ÉVITER entre 0 et 100,
  "summary": "En deux phrases, résume les arguments les plus forts de chaque camp avec les chiffres qui vont avec",
  "arguments": "Cite LES DEUX chiffres ou LES DEUX faits précis qui ont emporté ta décision"
}

RÈGLE D'OR : Ne fais pas de résumé poli. Tranche comme une guillotine. Ne raconte pas ta vie on a pas de temps pour ca, il n'y a que le proft a long terme qui nous interesse.
"""


tool = TavilySearch(max_results=4)
# 4. Construction de la chaîne (Pipeline)
# Le pipe "|" lie les composants : Prompt -> Modèle -> Nettoyage du texte
agent_optimiste = Agent(model, [tool], system=prompt_bull)
agent_pessimiste = Agent(model, [tool], system=prompt_bear)
agent_moderateur = Agent(model, [tool], system=prompt_moderateur)

# 1. Configuration de la page Streamlit
st.set_page_config(page_title="IA Investisseur", page_icon="📈")
st.title("🚀 Agent d'Investissement Optimiste")
entreprise = st.text_input("Entrez le nom d'une entreprise :", placeholder="ex: NVIDIA, LVMH, Tesla...")

if st.button("Lancer l'investigation"):
    if entreprise:
        # Création de deux colonnes pour l'affichage côte à côte
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🚀 L'Avis du Bull")
            with st.spinner("L'optimiste cherche des arguments..."):
                res_opt = agent_optimiste.graph.invoke({"messages": [HumanMessage(content=f"Pourquoi investir dans {entreprise} ?")]})
                st.success(res_opt['messages'][-1].content)
                
        with col2:
            st.header("📉 L'Avis du Bear")
            with st.spinner("Le sceptique cherche les failles..."):
                res_pess = agent_pessimiste.graph.invoke({"messages": [HumanMessage(content=f"Quels sont les risques d'investir dans {entreprise} ?")]})
                st.error(res_pess['messages'][-1].content)
        st.divider()
        st.header("⚖️ VERDICT DU MODÉRATEUR (Self-Consistency)")
        with st.spinner("⚖️ Le CIO compare les deux positions et va rendre sa décision..."):
            debat_txt = f"BULL: {res_opt['messages'][-1].content}\nBEAR: {res_pess['messages'][-1].content}"
            final = get_final_decision(debat_txt)
            
            if final:
                # On détermine si c'est un succès (Achat) ou une erreur (Éviter)
                is_achat = "ACHAT" in final['verdict'].upper()
                score_confiance = final['achat'] if is_achat else final['eviter']
                if is_achat:
                    st.success(f"### ✅ {final['verdict']}  |  Score de certitude : {score_confiance:.1f}%")
                else:
                    st.error(f"### ❌ {final['verdict']}  |  Score de certitude : {score_confiance:.1f}%")
            
            st.info(f"**RESUME**\n\n{final['explication']}")
                
            st.warning(f"**FACTEUR DÉCLIC :** {final['declic']}")
    else:
        st.warning("Veuillez entrer un nom d'entreprise.")
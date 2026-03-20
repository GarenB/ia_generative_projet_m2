import json
import numpy as np
import streamlit as st
import operator
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq # Remplacement de OpenAI par Groq (Llama)
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

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

model_llama = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)
tool = TavilySearchResults(max_results=2)

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
Tu es un trader de hedge fund agressif. Ton but est le profit à long terme (5 ans apres l'achat).

TA MISSION : Comparer les deux analyses et trancher si il faut ACHETER ou ÉVITER.

RÈGLES DE DÉCISION :
1. Si le potentiel de gain (Bull) est massivement supérieur aux risques (Bear), choisis ACHAT même si le Bear a des arguments.
2. Tranche de manière BRUTALE, PRECISE et BINAIRE.

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

RÈGLE D'OR : Ne fais pas de résumé poli. Tranche comme une guillotine. Ne raconte pas ta vie on a pas de temps pour ca.
"""


agent_optimiste = Agent(model_llama, [tool], system=prompt_bull)
agent_pessimiste = Agent(model_llama, [tool], system=prompt_bear)
agent_moderateur = Agent(model_llama, [], system=prompt_moderateur)

st.set_page_config(page_title="IA Investisseur Llama", page_icon="📈")
st.title("📈 Llama Finance Investigator")
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
    with st.spinner("⚖️ Décision du CIO en cours..."):
        debat_txt = f"BULL: {res_opt['messages'][-1].content}\nBEAR: {res_pess['messages'][-1].content}"
        final = get_final_decision(debat_txt)
        
        is_achat = "ACHAT" in final['verdict'].upper()
        score = final['achat'] if is_achat else final['eviter']
        
        if is_achat:
            st.success(f"### ✅ {final['verdict']} | Certitude : {score:.1f}%")
            st.error(f"### ❌ {final['verdict']} | Certitude : {score:.1f}%")
        else:
            st.error(f"### ❌ {final['verdict']} | Certitude : {score:.1f}%")
            st.success(f"### ✅ {final['verdict']} | Certitude : {score:.1f}%")
        
        st.info(f"**RÉSUMÉ :** {final['explication']}")
        st.warning(f"**DÉCLIC :** {final['declic']}")
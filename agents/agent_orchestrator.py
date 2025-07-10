# Main agent class containing the logic of running complete flow

import json
import re
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage , AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from agents.Drug_Analysis.chatbot import MedicalChatbot
from agents.Intent_Analysis.intent_analysis import IntentIdentifier
from agents.Medical_Analysis.Medical_rag import MedicalAgent
from agents.ResponderAgent.responderAgent import ResponsderAgent
from agents.Utils.common_methods import extract_image_info

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    finalResponse : str = ""
    intent : str

# Agents Method 


def intentAgent(state : AgentState ) :
    last_msg = state.messages[-1]['content'] 
    agentIntent = IntentIdentifier(state)
    intent_response = agentIntent.get_intent_agent_response(last_msg)[1]
    pattern = r'\{.*?\}'
    match = re.search(pattern,intent_response, re.DOTALL)
    if match:
        json_str = match.group()
        intent_result = json.loads(json_str)

    intent = intent_result['actual_tag']
    return {**state , "intent" : intent}
    
    
def disease_agent(state: AgentState):
    query = state.messages[-1]['content']
    diseaseAgent = MedicalChatbot(state)
    disease_response = diseaseAgent.process_user_message(query)[0]
    # state["messages"] = AIMessage(content=disease_response)
    return {**state, "finalResponse": disease_response}


def drugs_agent(state: AgentState):
    query = state.messages[-1]['content']
    image_info = extract_image_info(query)
    drugsAgent = MedicalAgent()
    drugs_response = drugsAgent.get_responder_output(isImage=image_info.get("isImage"), image_source=image_info.get("imageSource"), query=query)
    # state["messages"] = AIMessage(content=drugs_response)
    return {**state, "finalResponse": drugs_response}

def responder_agent(state: AgentState):
    query = state.messages[-1]['content']
    intent = state.intent
    finalResponse = state.finalResponse
    responder_agent = ResponsderAgent(state)
    responder_agent_response = responder_agent.get_responder_output(user_query=query, intent=intent , final_response=finalResponse)
    state["messages"] = {"role" : "Assistant" , "content" : responder_agent_response}
    return {**state , "finalResponse" : responder_agent_response}

def intent_condition(state : AgentState):
    return state["intent"]

def graph_compilation():
    graph = StateGraph(AgentState)
    graph.add_node("intent", intentAgent)
    graph.add_node("disease_agent", disease_agent)
    graph.add_node("drug_agent", drugs_agent)
    graph.add_node("responder_agent", responder_agent)

    graph.set_entry_point("intent")
    graph.add_conditional_edges("intent",intent_condition, {
        "disease_and_symptom_analyzer": "disease_agent",
        "drugs_analyser": "drug_agent",
        "small_talk" : "responder_agent"
    })
    graph.add_edge("disease_agent", "responder_agent")
    graph.add_edge("drug_agent", "responder_agent")
    graph.set_finish_point("responder_agent")
    app_graph = graph.compile()
    return app_graph
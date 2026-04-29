from dotenv import load_dotenv
from typing import TypedDict, Literal, Final
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path
import re
import json

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#Shared Multi-Agent State
class MultiAgentState(TypedDict):
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user


#read from yaml
with open("prompts/supervisor_v1.yaml", "r") as f:
    supervisor_system_prompt_from_yaml = f.read()

#Supervisor prompt from Yaml
VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

#SuperVisor Node
def supervisor_node(state: MultiAgentState) -> dict:
    messages = [
        SystemMessage(content=supervisor_system_prompt_from_yaml),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}

#Route to Specialist
def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")


#Create Tools
@tool
def orders_agent(user_query: str,order_id: str):
    """Use this tool to get order status of the customer
    
    Parameters:
    user_query (str): The query from the user
    order_id (str): The order ID of the customer
    Returns: 
    str: The order status of the customer after cross checking with the database.
    """
    return "Order status: " + user_query

@tool
def billing_agent(user_query: str,customer_id: str):
    """Use this tool to get billing information of the customer
    
    Parameters:
    user_query (str): The query from the user
    customer_id (str): The customer ID of the customer
    Returns: 
    str: The billing information of the customer after cross checking with the database.
    """
    return "Billing information: " + user_query

@tool
def technical_agent(user_query: str,customer_id: str):
    """Use this tool to resolve technical issues of the customers
    
    Parameters:
    user_query (str): The query from the user
    customer_id (str): The customer ID of the customer
    Returns: 
    str: The technical information of the customer after cross checking with the database.
    """
    return "Technical information: " + user_query

@tool
def subscription_agent(user_query: str,customer_id: str):
    """Use this tool to get subscription information of the customers
    
    Parameters:
    user_query (str): The query from the user
    customer_id (str): The customer ID of the customer
    Returns: 
    str: The subscription information of the customer after cross checking with the database.
    """
    return "Subscription information: " + user_query

@tool
def general_agent(user_query: str):
    """Use this tool to answer general questions
    
    Parameters:
    user_query (str): The query from the user
    Returns: 
    str: The general information of the customer after cross checking with the database.
    """
    return "General information: " + user_query

def orders_agent_node(state: MultiAgentState) -> dict:
    """Call the orders tool and return the result
    
    Parameters:
    state (MultiAgentState): The state of the graph
    Returns: 
    dict: The result of the orders tool
    """
    text = f"[orders_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "orders_agent",
        "specialist_result": text,
    }

def billing_agent_node(state: MultiAgentState) -> dict:
    """Call the billing tool and return the result
    
    Parameters:
    state (MultiAgentState): The state of the graph
    Returns: 
    dict: The result of the billing tool
    """
    text = f"[billing_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "billing_agent",
        "specialist_result": text,
    }

def technical_agent_node(state: MultiAgentState) -> dict:
    """Call the technical tool and return the result
    
    Parameters:
    state (MultiAgentState): The state of the graph
    Returns: 
    dict: The result of the technical tool
    """
    text = f"[technical_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "technical_agent",
        "specialist_result": text,
    }

def subscription_agent_node(state: MultiAgentState) -> dict:
    """Call the subscription tool and return the result
    
    Parameters:
    state (MultiAgentState): The state of the graph
    Returns: 
    dict: The result of the subscription tool
    """
    text = f"[subscription_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "subscription_agent",
        "specialist_result": text,
    }

def general_agent_node(state: MultiAgentState) -> dict:
    """Call the general tool and return the result
    
    Parameters:
    state (MultiAgentState): The state of the graph
    Returns: 
    dict: The result of the general tool
    """
    text = f"[general_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "general_agent",
        "specialist_result": text,
    }

def synthesize_response_node(state: MultiAgentState) -> dict:
    response = f"Thank you for contacting us. Your request has been handled by our {state['route']} team. {state['specialist_result']}"
    return {"final_response": response}

def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")

    workflow.add_conditional_edges(
        "supervisor_node",
        route_to_specialist,
    )

    for specialist in [
        "orders_agent_node",
        "billing_agent_node",
        "technical_agent_node",
        "subscription_agent_node",
        "general_agent_node",
    ]:
        workflow.add_edge(specialist, "synthesize_response")

    workflow.add_edge("synthesize_response", END)

    return workflow.compile()

@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str   # "low" | "normal" | "high"
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {self.context}\n"
            f"Received at: {self.timestamp}"
        )

# Example usage (commented out to prevent NameError):
# handoff = AgentHandoff(
#     from_agent="supervisor",
#     to_agent="billing",
#     task=state["user_request"],
#     context={"route": state["route"]},
#     priority="normal",
#     timestamp=datetime.utcnow().isoformat(),
# )

INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (your |all |previous )?instructions",
    r"system prompt.*disabled",
    r"you are now a",
    r"repeat.*system prompt",
    r"jailbreak",
]

def detect_injection(user_input: str) -> bool:
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "I can only assist with account and order support. (Request blocked.)"
    return user_input

@dataclass
class SessionAuditLog:
    session_id: str
    events: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def log(self, agent: str, action: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        cost = (tokens_in * 0.000015 + tokens_out * 0.00006) / 1000
        self.total_cost_usd += cost
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "action": action,
                "cost_usd": round(cost, 6),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "events": self.events,
        }

def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path(__file__).parent.parent / "audit_log.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")

def main() -> None:
    audit = SessionAuditLog(session_id="demo-session")
    graph = build_graph()

    for request in [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
    ]:
        safe_text = guard_request(request)
        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
        }
        result = graph.invoke(state)
        print("Request:", request)
        print("Route:", result.get("route"), "Agent used:", result.get("agent_used"))
        print("Final:", result.get("final_response"))
        print("---")

    print("Total cost (USD):", round(audit.total_cost_usd, 6))
    persist_audit_log(audit)


if __name__ == "__main__":
    main()
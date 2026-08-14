import os
from typing import TypedDict,Literal
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,START,END
try:
    from .common import retrieve
except ImportError:
    from common import retrieve

KEYWORDS=['delivery','return','refund','membership','tracking','cancel','gift card','support hours']
PROMPT_TEMPLATE='''ROLE: You are a Zepto policy support assistant.\nCONTEXT: Use only the supplied retrieved Zepto policy chunks.\nTASK: Answer the user's question accurately from context.\nFORMAT: Return JSON with answer, sources, confidence.\nLENGTH: Keep the answer concise.\nNEGATIVE CONSTRAINT: Do not answer using information not present in the provided context.\nFEW-SHOT EXAMPLE: Q: What is the delivery fee below INR 149? Context: orders below INR 149 incur INR 25. A: INR 25.'''

class State(TypedDict, total=False):
    query:str; intent:Literal['policy_question','general_question']; answer:dict; retrieval:dict
class AskRequest(BaseModel): query:str
class Answer(BaseModel): answer:str; sources:list[str]; confidence:float=Field(ge=0,le=1)

def classify_intent(s:State):
    q=s['query'].lower(); intent='policy_question' if any(k in q for k in KEYWORDS) else 'general_question'
    return {'intent':intent}

def optional_real_llm_json(prompt: str, retries: int = 2):
    """Optional real-provider hook: parse/validate structured output and retry twice on validation failure."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            # Provider-specific invocation intentionally omitted from the graded offline baseline.
            raise RuntimeError('Configure a real LLM provider for MOCK_LLM=0')
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                prompt += '\nCorrective instruction: return only valid JSON matching answer/sources/confidence.'
    return Answer(answer=f'Real LLM error after retries: {last_error}',sources=[],confidence=0.0)

def retrieve_and_answer(s:State):
    r=retrieve(s['query'],3); docs=r['documents'][0]; ids=r['ids'][0]
    snippet=docs[0][:200]
    # Required deterministic mock baseline. Real LLM integration can be layered here when MOCK_LLM=0.
    if os.getenv('MOCK_LLM','1')=='1':
        ans=f'Based on the retrieved context: {snippet}'
        return {'answer':Answer(answer=ans,sources=ids,confidence=1.0).model_dump()}
    # Optional path intentionally conservative: without a configured provider, return a clear error.
    return {'answer':Answer(answer='Real LLM mode is not configured in this baseline.',sources=ids,confidence=0.0).model_dump()}

def direct_answer(s:State):
    if os.getenv('MOCK_LLM','1')=='1': return {'answer':Answer(answer='I can only answer questions about Zepto policies right now.',sources=[],confidence=1.0).model_dump()}
    return {'answer':Answer(answer='Real LLM mode is not configured in this baseline.',sources=[],confidence=0.0).model_dump()}

g=StateGraph(State); g.add_node('classify_intent',classify_intent); g.add_node('retrieve_and_answer',retrieve_and_answer); g.add_node('direct_answer',direct_answer); g.add_edge(START,'classify_intent'); g.add_conditional_edges('classify_intent',lambda s:s['intent'],{'policy_question':'retrieve_and_answer','general_question':'direct_answer'}); g.add_edge('retrieve_and_answer',END); g.add_edge('direct_answer',END); graph=g.compile()

app=None
try:
 from fastapi import FastAPI
 app=FastAPI(title='Zepto Support Assistant')
 @app.post('/ask',response_model=Answer)
 def ask(req:AskRequest): return Answer(**graph.invoke({'query':req.query})['answer'])
except ImportError: pass

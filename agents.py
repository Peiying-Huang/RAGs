from openai import OpenAI
from typing import Type
from pydantic import BaseModel
import os
from system_prompts import SystemPrompts
from dotenv import load_dotenv
from typing import Type, List
#from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
#from openai import RateLimitError

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API"))


class QueryDecomposeOutput(BaseModel):
    original_query: str
    decomposed_queries: list[str]


class QueryParaphraseOutput(BaseModel):
    original_query: str
    paraphrased_queries: list[str]


class StepBackOutput(BaseModel):
    original_query: str
    stepback_queries: list[str]

class EvaluatorOutput(BaseModel):
    relevance: str 

class JudgeOutput(BaseModel):
    sufficiency: str
    strategy: str
    
class MasterOutput(BaseModel):
    agent: str

class GenerationOutput(BaseModel):
    answer: str
    new_search_query:str

class Agents:
    def __init__(self, client:str = client, model: str = "gpt-5-mini-2025-08-07"):
        self.client = client
        self.model = model

    def run(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]): #, temperature: float = 0.8) -> BaseModel:

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            text_format=output_schema
        )

        return response.output_parsed
    
    
    def decompose(self, query: str, num_queries: int =3): #, temperature: float = 0.8):

        return self.run(SystemPrompts.QUERY_DECOMPOSE(num_queries), query,QueryDecomposeOutput)#, temperature=temperature)

    def paraphrase(self, query: str, num_queries:int =3): #, temperature: float = 0.8):

        return self.run(SystemPrompts.QUERY_PARAPHRASE(num_queries), query, QueryParaphraseOutput) #,, temperature=temperature)

    def stepback(self, query: str, num_queries:int =3): #,,temperature: float = 0.8):

        return self.run(SystemPrompts.QUERY_STEPBACK(num_queries), query, StepBackOutput) #,, temperature=temperature)

    def evaluate(self, query: str, article_dict:dict): #, temperature: float = 0.8)
        
        return self.run(SystemPrompts.EVALUATOR(),SystemPrompts.EVALUATOR_USER(query, article_dict), EvaluatorOutput)#,temperature=temperature)

    def master(self, query: str): #, temperature: float = 0.8):

        return self.run(SystemPrompts.MASTER(),query, MasterOutput) #, temperature=temperature)

    def judge(self, query: str, all_documents:list, iteration: int): #,, temperature: float = 0.8):
        
        return self.run(SystemPrompts.JUDGE(),SystemPrompts.JUDGE_USER(query, all_documents, iteration) , JudgeOutput)#,temperature=temperature)

    def generate(self, query: str,retrieved_results: List[str]): #,, temperature: float = 0.8):
        
        return self.run(SystemPrompts.GENERATION(),SystemPrompts.GENERATION_USER(query, retrieved_results), GenerationOutput)
    
"""
@retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(min=1, max=8),
        stop=stop_after_attempt(3),
    )
"""

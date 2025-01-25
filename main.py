from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
import json
import pandas as pd
import numpy as np

print("Starting FastAPI application...")
app = FastAPI()

class Message(BaseModel):
    user_id: Optional[int]
    message: str

def initialize_csv():
    attributes = ['kindness', 'patience', 'honesty', 'humor', 'empathy', 
                 'emotional_intelligence', 'communication_skills', 'conflict_resolution',
                 'trustworthiness', 'loyalty', 'ambition', 'drive', 'perseverance',
                 'adaptability', 'open_mindedness', 'curiosity', 'introversion',
                 'extroversion', 'social_skills', 'leadership', 'independence',
                 'creativity', 'adventurousness', 'discipline', 'responsibility',
                 'time_management', 'generosity']
    
    df = pd.DataFrame(0, index=['contestant_1', 'contestant_2'], columns=attributes)
    df.to_csv('contestant_scores.csv')
    return df

def update_csv_scores(contestant_id: int, scores: Dict):
    df = pd.read_csv('contestant_scores.csv', index_col=0)
    contestant_row = f'contestant_{contestant_id}'
    for attr, score in scores.items():
        df.loc[contestant_row, attr] = score
    df.to_csv('contestant_scores.csv')
    return df

def calculate_winner(df: pd.DataFrame) -> Dict:
    personality_traits = ['kindness', 'patience', 'honesty', 'humor', 'empathy']
    emotional_traits = ['emotional_intelligence', 'communication_skills', 'conflict_resolution']
    reliability_traits = ['trustworthiness', 'loyalty', 'responsibility']
    drive_traits = ['ambition', 'drive', 'perseverance']
    
    weights = {
        'personality': 0.3,
        'emotional': 0.25,
        'reliability': 0.25,
        'drive': 0.2
    }
    
    scores = {}
    for contestant in df.index:
        personality_score = df.loc[contestant, personality_traits].mean()
        emotional_score = df.loc[contestant, emotional_traits].mean()
        reliability_score = df.loc[contestant, reliability_traits].mean()
        drive_score = df.loc[contestant, drive_traits].mean()
        
        final_score = (personality_score * weights['personality'] +
                      emotional_score * weights['emotional'] +
                      reliability_score * weights['reliability'] +
                      drive_score * weights['drive'])
        
        scores[contestant] = final_score
    
    winner_id = int(max(scores.items(), key=lambda x: x[1])[0].split('_')[1])
    reason = f"Contestant {winner_id} scored higher in weighted attributes evaluation"
    
    return {"winner_id": winner_id, "reason": reason}

class GameState:
    def __init__(self):
        self.current_round = 0
        self.max_rounds = 1
        self.contestants = {1: "Contestant 1", 2: "Contestant 2"}
        self.questions_asked = []
        self.current_question = None
        self.awaiting_responses = False
        self.attribute_scores = {}
        self.df = initialize_csv()
        print("GameState initialized with CSV file")

game_state = GameState()

print("Initializing Groq LLM...")
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

def update_attributes(contestant_id: int, answer: str) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the contestant's answer and rate these attributes (1-10).
        kindness, patience, honesty, humor, empathy, emotionalintelligence, 
        communicationskills, conflict_resolution, trustworthiness, loyalty,
        ambition, drive, perseverance, adaptability, open_mindedness,
        curiosity, introversion, extroversion, socialskills, leadership,
        independence, creativity, adventurousness, discipline, responsibility,
        timemanagement, generosity. 
         
        Choose ONLY exactly the 3 most relevant attributes to the answer of the contestant.
         

        
        Return ONLY a JSON object with the exact names of these attributes as provided to you along with numerical scores. No explanations."""),
        ("human", f"Answer: {answer}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({'answer': answer})
    print("Response: json", response['text'])

    
    try:
        # print response
        response_dict = json.loads(response['text'])
        print("Parsed attributes:", response_dict)
        return response_dict
    except:
        print("Parsing failed for response:", response)
        return {}

def generate_question() -> str:
    print(f"Generating question for round {game_state.current_round}")
    print(f"Previous questions: {game_state.questions_asked}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a meaningful dating show question."),
        ("human", f"Previous questions: {game_state.questions_asked}. Round: {game_state.current_round}/1")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    print("Sending request to Groq for question generation...")
    response = chain.invoke({})
    print(f"Generated question: {response}")
    return response['text']

print("Initializing tools...")
tools = [
    Tool(name="update_attributes", func=update_attributes, description="Updates contestant attributes"),
    Tool(name="generate_question", func=generate_question, description="Generates a question")
]

print("Setting up memory and agent...")
memory = ConversationBufferWindowMemory(k=10)
agent = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory, verbose=True)

@app.post("/chat")
async def chat(message: Message):
    print(f"\nReceived chat message: {message}")
    
    if message.user_id:
        if message.user_id not in [1, 2]:
            return {"response": "Invalid contestant ID. Only contestants 1 and 2 are allowed."}
            
        if game_state.awaiting_responses:
            attributes = update_attributes(message.user_id, message.message)
            print(f"Attributes extracted: {attributes}")
            game_state.attribute_scores[message.user_id]= pd.read_csv('contestant_scores.csv', index_col=0).loc[f'contestant_{message.user_id}'].to_dict()
            for key, value in attributes.items():
                game_state.attribute_scores[message.user_id][key] = value


                

            print(f"Current scores: {game_state.attribute_scores}")

            game_state.df = update_csv_scores(message.user_id, attributes)
            # save scores to CSV
            game_state.df.to_csv('contestant_scores.csv')
            print(f"Updated scores: {game_state.attribute_scores}")
            
            if len(game_state.attribute_scores) == 2:  # Both contestants answered
                winner = calculate_winner(game_state.df)
                return {"response": f"Game Over! Winner: {winner['winner_id']}. Reason: {winner['reason']}"}
            
            return {"response": "Answer recorded. Waiting for other contestant..."}
        
        return {"response": "Please wait for the question."}
    
    else:
        if "start" in message.message.lower():
            game_state.attribute_scores = {}
            game_state.df = initialize_csv()
            question = generate_question()
            game_state.current_question = question
            game_state.awaiting_responses = True
            return {"response": f"Game started! Question: {question}"}
            
        return {"response": agent.run(message.message)}
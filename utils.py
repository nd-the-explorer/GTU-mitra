from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
import openai
import streamlit as st
openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

# os.environ['PINECONE_API_KEY'] = '7aa73978-ba4f-4f7b-adb3-eeb6a425c3f6'
os.environ['PINECONE_API_ENV'] = 'gcp-starter'

pc = Pinecone(api_key='PINECONE_API_KEY')
index = pinecone.Index(api_key='7aa73978-ba4f-4f7b-adb3-eeb6a425c3f6', index_name='chatbot', host="https://chatbot-g8yjyor.svc.gcp-starter.pinecone.io")

def find_match(input):
    input_em = model.encode(input).tolist()
    
    if input_em is None:
        return "Error: Model encoding returned None"
    
    result = index.query(vector=input_em, top_k=2, includeMetadata=False)
    
    if result is None or 'matches' not in result or len(result['matches']) < 2:
        return "Less than 2 matches found"
    
    match1_text = result['matches'][0].text
    match2_text = result['matches'][1].text
    
    if match1_text is None or match2_text is None:
        return "Error: Match text is None"
    
    return match1_text + "\n" + match2_text




# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(vector=input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# def query_refiner(conversation, query):

#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


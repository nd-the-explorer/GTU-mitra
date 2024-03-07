from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
import openai
import streamlit as st
model = SentenceTransformer('all-MiniLM-L6-v2')

# os.environ['PINECONE_API_KEY'] = ''
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']

pc = Pinecone(api_key='PINECONE_API_KEY')
index = pinecone.Index(api_key=st.secrets['OPENAI_API_KEY'], index_name='chatbot', host=st.secrets['PINECONE_HOST'])

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


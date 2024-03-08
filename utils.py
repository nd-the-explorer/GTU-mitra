from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
import openai
import streamlit as st
from dotenv import load_dotenv
load_dotenv(r'C:\Users\darji\OneDrive\Desktop\chatbot\.env')

openai.api_key = os.getenv('OPENAI_API_KEY')
model = SentenceTransformer('all-MiniLM-L6-v2')


pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pinecone.Index(api_key=os.getenv('PINECONE_API_KEY'), index_name=os.getenv('PINECONE_INDEX_NAME'), host=os.getenv('PINECONE_HOST'))

# def find_match(input):
#     input_em = model.encode(input).tolist()   
#     if input_em is None:
#         return "Error: Model encoding returned None"   
#     result = index.query(vector=input_em, top_k=2, includeMetadata=False)   
#     if result is None or 'matches' not in result or len(result['matches']) < 2:
#         return "Less than 2 matches found"    
#     match1_text = result['matches'][0].text
#     match2_text = result['matches'][1].text    
#     if match1_text is None or match2_text is None:
#         return "Error: Match text is None"    
#     return match1_text + "\n" + match2_text

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    # Check if there are matches
    if 'matches' in result:
        # Iterate through the matches
        matches_info = []
        for match in result['matches']:
            metadata = match.get('metadata', {})
            text=metadata.get('text','')
            datetime_info = metadata.get('datetime', '')
            page_info = metadata.get('page', '')
            source_info = metadata.get('source', '')
            match_info = f"Date and Time: {datetime_info}, Page: {page_info}, Source: {source_info}, text: {text}"
            matches_info.append(match_info)
        
        return "\n".join(matches_info)
    else:
        return "No matches found"


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

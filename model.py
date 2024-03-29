from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import os
from dotenv import load_dotenv
load_dotenv(r'C:\Users\darji\OneDrive\Desktop\chatbot\.env')

st.header("GTU Mitra- AI based chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name=os.getenv('MODEL'), openai_api_key=os.getenv('OPENAI_API_KEY'))
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(
    template=
    """
    If user greets you, then greet user back.

    Only answer the question if it is present in the context,
    don't try to generate answer if it is not present in the context. 

    When processing queries related to specific information which can change over a period of time, 
    such as name of someone on a specific position or a contact details or etc, 
    it's essential to consider the datetime information associated with each piece of context chunk to ensure accuracy and relevance.

    1. In the context you are provided with the different chunks each having datetime information and the relevant inforamtion.
    2. considering the datetime information, Select the part of the context with the latest datetime.
    3. Based on the selected context chunk, provide the relevant response. 

    If you don't know the answer then politely say sorry and say that you don't have that information currently.

    
    
    """)


#Prompt 1: Answer the question as truthfully as possible using the provided context,and if the answer is not contained within the text below, say 'I don't know'
#problem: When asked about IIT delhi it gives proper response.

#Prompt 2: Only answer the question if it is present in the following text,don't try to generate answer if it is not present in the text. If you don't know the answer then politely say sorry and say that you don't have that information currently""")
#problem: says dont know even when user says good morning.

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)



# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.chat_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            # refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            context = find_match(query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        

     
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')



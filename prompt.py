from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
st.header('Research Tool')
# user_input = st.text_input('Enter your prompt')

paper_input = st.selectbox('Select Research Paper Name',['Attention is all you need',
'bert: Pre-traning of deep bidirectional Transformers','GPT-3: Language Models are few-Shot Learners',
'Diffusion models beat GANS on image synthesis'])
style_input = st.selectbox('Select explaination style',['Beginner-Friendly','Technical','code oriented',
'Mathematical'])

length_input = st.selectbox('Select  explanation length',['Short (1-2 paragraph)','medium(3-4 pargraph)',
'long(detailed explanation)'])

template = PromptTemplate (                      
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications: 
Explanation Style: {style_input} 
Explanation Length: {length_input} 
1. Mathematical Details: 
Include relevant mathematical equations if present in the paper. 
Explain the mathematical concepts using simple, intuitive code snippets where applicable. 
2. Analogies: 
Use relatable analogies to simplify complex ideas. 
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing. 
Ensure the summary is clear, accurate, and aligned with the provided style and length. """,
input_variables=['paper_input', 'style_input', 'length_input'], 
validate_template=True)

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input':length_input
    
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)

#input_variable = ['paper_input','style_input','length_input']
#st.header('Research Paper')
# if st.button:
#     st.text('some random Text')
#     result = model.invoke(user_input)
#     st.write(result.content)
    
    
# if st.button('Summarize'):
#     result = model.invoke(user_input)
#     st.write(result.content)
    
    
    

# result = model.invoke("what is the capiital of india??")
# print(result.content)
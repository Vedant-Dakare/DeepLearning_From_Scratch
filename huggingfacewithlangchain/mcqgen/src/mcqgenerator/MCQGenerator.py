import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GROQ_KEY")

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.5
)

template="""
Text:{text}
Act as an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""


quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template,
    verbose=True
    )

from langchain_core.runnables import RunnablePassthrough

quiz_chain = (
    quiz_generation_prompt
    | llm
)

template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template2, verbose=True)

review_chain = quiz_evaluation_prompt | llm


## combining the chains together
from langchain_core.runnables import RunnableLambda

generate_evaluate_chain = (
    RunnableLambda(
        lambda x: {
            "quiz_result": quiz_chain.invoke(x),
            "subject": x["subject"]
        }
    )
    | RunnableLambda(
        lambda x: {
            "quiz": x["quiz_result"].content,
            "review": review_chain.invoke({
                "quiz": x["quiz_result"].content,
                "subject": x["subject"]
            }).content
        }
    )
)

# need some work to be done 

from agentic_chunker import AgenticChunker
from imoprt_data import load_pdf

# https://arxiv.org/pdf/2312.06648.pdf

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub
import json

obj = hub.pull("wfh/proposal-indexing")
llm = OllamaLLM(model="llama3.2:latest")
runnable = obj | llm

class Sentences(BaseModel):
    sentences: List[str]
# Extraction
extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
def get_propositions(text):

    json_data = runnable.invoke({
    	"input": text
    })

    runnable_output  = json.loads(json_data) # this will not work it need to be parsed to use it has some junk before the json starts 

    propositions = extraction_chain.invoke(runnable_output)["text"][0].sentences
    return propositions

text = load_pdf("R22_syllabus.pdf")
    
paragraphs = text.load_and_split()
text_propositions = []
for i, para in enumerate(paragraphs[:5]):
    propositions = get_propositions(para)
    text_propositions.extend(propositions)
    print (f"Done with {i}")

print (f"You have {len(text_propositions)} propositions")
print(text_propositions[:10])

ac = AgenticChunker("llama3.2:1b")
ac.add_propositions(text_propositions)
print(ac.pretty_print_chunks())
chunks = ac.get_chunks(get_type='list_of_strings')
print(chunks)
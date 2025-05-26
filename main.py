from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_and_split_docs
from transformers import pipeline

load_dotenv()

docs = load_and_split_docs("data/sample_doc.txt")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents=docs, embedding=embedding_model, persist_directory="chroma_db")

# flan-t5-base model
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=300,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

llm = HuggingFacePipeline(pipeline=pipe)

# Chain 1: QA from Vector DB
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chain 2: Simple Explanation Chain
explain_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain this in simple terms:\n\n{topic}"
)
llm_chain = LLMChain(llm=llm, prompt=explain_prompt)

# Chain 3: Summary Chain
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following information:\n\n{text}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Chain 4: Conversational Memory Chain
conversation_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="Continue the conversation: \n\n{history}\nUser: {input}\nAssistant:"
)
memory = ConversationBufferMemory()
convo_chain = LLMChain(llm=llm, prompt=conversation_prompt, memory=memory)

print(" LangChain Research Assistant is ready!")

while True:
    user_input = input("\nAsk a question (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    print("\n Answer from QA Chain:")
    print(qa_chain.invoke(user_input)["result"])

    print("\n Simplified Explanation:")
    print(llm_chain.invoke({"topic": user_input})["text"])

    print("\n Summary Style:")
    print(summary_chain.invoke({"text": user_input})["text"])

    print("\n Conversation Response:")
    print(convo_chain.invoke({"input": user_input})["text"])

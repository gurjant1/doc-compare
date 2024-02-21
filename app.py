from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field, BaseModel
from langchain.agents import initialize_agent, AgentType
from flask import Flask, request, jsonify
from typing import Any
import os
from dotenv import load_dotenv
from langchain.globals import set_debug
from slugify import slugify
set_debug(True)
load_dotenv()
app = Flask(__name__)


class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                 openai_api_key=os.environ.get("OPENAI_API_KEY"))


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both file1 and file2 must be provided'})

    file1 = request.files['file1']
    file2 = request.files['file2']

    os.remove("uploads/*")

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Both file1 and file2 must be selected'})

    filename1 = ("uploads/" + file1.filename)

    filename2 = (
        "uploads/" + file2.filename
    )

    file1.save(filename1)
    file2.save(filename2)

    return jsonify({'message': 'Files saved successfully'})


tools = []
path = "uploads/"


def get_files_from_folder(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files.append({"name": filename, "path": file_path})
    return files


files = get_files_from_folder(path)

print(files)

for file in files:
    loader = PyPDFLoader(file["path"])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=slugify(file["name"]),
            description="Retrieve documents from uploaded PDF file",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )
agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)


@app.route('/compare_documents', methods=['POST'])
def compare_documents():
    data = request.get_json()

    if 'question' not in data:
        return jsonify({"error": "question field 'question' is required."}), 400

    result = agent({"input": data['question']})

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

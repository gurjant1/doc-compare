from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field, BaseModel
from langchain.agents import initialize_agent, AgentType
from flask import Flask, render_template, request, jsonify
from typing import Any
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from slugify import slugify

load_dotenv()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                 openai_api_key=os.environ.get("OPENAI_API_KEY"))


def get_files_from_folder(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files.append({"name": filename, "path": file_path})
    return files


#  store intialized agent in global variable
agent = None


@app.route('/upload', methods=['POST'])
def upload_files():
    print(request.files.lists, "list")
    if 'files' not in request.files:
        return jsonify({'error': 'No files were provided'})

    uploaded_files = request.files.getlist('files')
    print(uploaded_files)

    if len(uploaded_files) < 2:
        return jsonify({'error': 'At least two files must be selected'})

    # delete all existing files in uploads folder
    for file in os.listdir("uploads"):
        os.remove(os.path.join("uploads", file))

    for file in uploaded_files:
        print(file.filename)
        if file.filename == '':
            return jsonify({'error': 'All files must have a filename'})

        file.save(os.path.join("uploads", file.filename))

    tools = []
    path = "uploads"
    files = get_files_from_folder(path)

    for file in files:
        if file["name"].endswith('.pdf'):
            loader = PyPDFLoader(file["path"])

        elif file["name"].endswith('.csv'):
            loader = CSVLoader(file["path"])
        else:
            # Handle unsupported file types
            continue

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = loader.load_and_split(text_splitter=text_splitter)
        embeddings = OpenAIEmbeddings()
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()

        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=slugify(file["name"]),
                description="Retrieve documents from uploaded file",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            )
        )
    global agent
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    return jsonify({'message': 'Files saved successfully'})


@app.route('/compare_documents', methods=['POST'])
def compare_documents():
    data = request.get_json()
    print(data, "data")

    if 'question' not in data:
        return jsonify({"error": "question field 'question' is required."}), 400

    result = agent({"input": data['question']})

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, port=8000, host="0.0.0.0")

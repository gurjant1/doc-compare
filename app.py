from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from flask import Flask, render_template, request, jsonify
from langchain.vectorstores.chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate

set_debug(True)
load_dotenv()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                 openai_api_key=os.environ.get("OPENAI_API_KEY",))

# if existing db folder exists, use it, otherwise create a new one
if os.path.exists("db"):
    pass
else:
    os.mkdir("db")
presist_directory = "db"
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=presist_directory,
                  embedding_function=embeddings)


def get_files_from_folder(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files.append({"name": filename, "path": file_path})
    return files


@app.route('/upload', methods=['POST'])
def upload_files():
    print(request.files.lists, "list")
    if 'files' not in request.files:
        return jsonify({'error': 'No files were provided'})

    uploaded_files = request.files.getlist('files')
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
    path = "uploads"
    files = get_files_from_folder(path)

    for file in files:
        if file["name"].endswith('.pdf'):
            loader = PyPDFLoader(file["path"])
        elif file["name"].endswith('.csv'):
            loader = CSVLoader(file["path"])
        else:
            continue

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = loader.load_and_split(text_splitter=text_splitter)
        # implement Chroma
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=presist_directory)
        vectordb.persist()
    return jsonify({'message': 'Files saved successfully and indexed'})


@app.route('/compare_documents', methods=['POST'])
def compare_documents():
    data = request.get_json()
    if 'question' not in data:
        return jsonify({"error": "question field 'question' is required."}), 400

    retriever = vectordb.as_retriever()
    relevant_documents = retriever.get_relevant_documents(
        data["question"], kwargs={"k": 2})

    print("Relevant documents: ", relevant_documents)
    print("Question: ", data["question"])

# Use the QA chain to answer the question
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
            template="Please answer the following question based solely on the information provided in the documents. Do not use any external or general knowledge sources. Question: {question} \n Context: {context}",
            input_variables=["question", "context"],
        )},
    )
    query = data["question"]
    answer = qa_chain(
        {"query": query, "source_documents": relevant_documents},
    )
    print("Answer: ", answer["result"])
    return jsonify({"output": answer["result"]}), 200


if __name__ == '__main__':
    app.run(debug=False, port=8000, host="0.0.0.0")

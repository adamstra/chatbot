import os
import yaml
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from llama_index.core import GPTVectorStoreIndex, Document
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# Classe pour créer des embeddings avec HuggingFace
class HuggingFaceEmbedding:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Utiliser HuggingFaceEmbedding
embedding_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")


# Charger les données YAML
def charger_donnees_yaml(dossier, fichiers_yaml):
    donnees = []
    for fichier in fichiers_yaml:
        chemin_fichier = os.path.join(dossier, fichier)
        with open(chemin_fichier, "r", encoding="utf-8") as f:
            contenu = yaml.safe_load(f)
            categories = contenu.get("categories", {})
            for categorie, conversations in categories.items():
                for conversation in conversations:
                    question, reponse = conversation
                    donnees.append((question, reponse))
    return donnees


# Dossier contenant les fichiers YAML
dossier_yaml = "data"
fichiers_yaml = [
    "der.yaml",
    "3fpt.yaml",
    "prodac.yaml",
    "fongip.yaml",
    "adepme.yaml",
]

# Charger et structurer les données
donnees_chatbot = charger_donnees_yaml(dossier_yaml, fichiers_yaml)

# Créer des documents et encoder les données YAML
documents = [
    Document(text=f"SenJobBot: {reponse}") for question, reponse in donnees_chatbot
]


# Fonction pour encoder les documents
def encode_documents(documents, embedding_model):
    encoded_docs = []
    for doc in documents:
        encoded_doc = embedding_model.encode(doc.text)
        encoded_docs.append((encoded_doc, doc))
    return encoded_docs


# Encoder les documents
encoded_documents = encode_documents(documents, embedding_model)


# Indexer les documents en utilisant une structure simple
class SimpleIndex:
    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add_document(self, embedding, document):
        self.embeddings.append(embedding)
        self.documents.append(document)

    def query(self, query_text):
        query_embedding = embedding_model.encode(query_text)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        best_match_idx = np.argmax(similarities)
        return self.documents[best_match_idx]


# Créer un index simple
index = SimpleIndex()
for embedding, document in encoded_documents:
    index.add_document(embedding, document)


# Endpoint pour poser une question
@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "La question est vide"}), 400

        document = index.query(question)
        return jsonify({"response": document.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Exemple d'utilisation locale avec Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

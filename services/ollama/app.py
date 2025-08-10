import os
import json
import requests
from flask import Flask, request, jsonify
from pinecone import Pinecone
import openai

app = Flask(__name__)

# === Environment Variables ===
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
ollama_url = "http://ollama:11434/api/generate"
PORT = int(os.environ.get("PORT", 5003))

index_name = "bible-verse-index"

# Create Pinecone index if not exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )
index = pc.Index(name=index_name)

# === Embedding helper ===
def embed_text(text):
    if not text.strip():
        return None
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"OpenAI Embedding error: {e}")
        return None

# === Routes ===
@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    try:
        with open("KJV.json", "r") as f:
            bible = json.load(f)

        upserts = []
        for book in bible["books"]:
            for chapter in book["chapters"]:
                for verse in chapter["verses"]:
                    text = verse["text"]
                    embedding = embed_text(text)

                    if not embedding or not any(embedding):
                        print(f"Skipping zero-vector verse: {verse['name']}")
                        continue

                    vec_id = f"{book['name']}_{chapter['chapter']}_{verse['verse']}"
                    upserts.append({
                        "id": vec_id,
                        "values": embedding,
                        "metadata": {
                            "book": book["name"],
                            "chapter": chapter["chapter"],
                            "verse": verse["verse"],
                            "text": text,
                            "name": verse["name"]
                        }
                    })

        batch_size = 100
        for i in range(0, len(upserts), batch_size):
            index.upsert(vectors=upserts[i:i + batch_size])
            print(f"Upserted batch {i // batch_size + 1}")

        return jsonify({"message": f"Successfully embedded {len(upserts)} verses."}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """Proxy to Ollama container inside Docker network"""
    try:
        r = requests.post(ollama_url, json=request.get_json(), timeout=300)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# === Run ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)

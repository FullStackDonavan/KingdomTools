import os
import openai
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec
import json

app = Flask(__name__)

openai.api_key = os.getenv("k-proj-LL5FmVsY5kq3fAmFQqnbL_LanGq8sIQpAKz8Dn1uW0h6EeZ8YzJKXx7QNPIWBKafwD25hfYKMyT3BlbkFJTx8i0mbgh8DMxWgU5GCoYcfNH4aVppjZtK6fihmEQyapGVIAmHUdyVsa25_kfvONNcpIjK_wsA")
pc = Pinecone(api_key=os.getenv("pcsk_6vVSon_Fy7EhEUgsmLngjHF5M29A4zJUDyBtijvvEc2uXfBaFbb4PA1Z1UpeX7EPzXmVKC"))
index_name = "bible-verse-index"

# Check if index already exists
if index_name not in Pinecone.list_indexes():
    Pinecone.create_index(
        name=index_name,
        dimension=1536,  # use your actual dimension
        metric="cosine"
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")
index = pc.Index(name=index_name)


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

if __name__ == "__main__":
    app.run(port=5003, debug=True)

import os
import json
from pathlib import Path
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIG ---
SUPABASE_URL = "https://mihrtlwzvlxoexlqiinf.supabase.co"
SUPABASE_KEY = "sb_secret_qDs57baY3h6PkBfbk17i5g_8CtL0164" # Use Service Role for bulk uploads
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Use the same model we used for Pinecone
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

def migrate_to_cloud():
    print("🚀 Starting Cloud Migration...")
    path = Path("./gita_parents_store")

    if not path.exists():
        print("⚠️ ./gita_parents_store not found; nothing to migrate.")
        return

    # Process each JSON file on your PC
    for file_path in path.glob("*.json"):
        # Skip empty files
        if file_path.stat().st_size == 0:
            print(f"⚠️ Skipping empty file {file_path.name}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping malformed JSON {file_path.name}")
            continue
        except Exception as e:
            print(f"⚠️ Error reading {file_path.name}: {e}")
            continue

        # Normalize to extract page_content and metadata
        page_content = None
        metadata = {}

        if isinstance(data, dict):
            if data.get("type") == "document":
                page_content = data.get("page_content")
                metadata = data.get("metadata", {})
            elif "page_content" in data:
                page_content = data.get("page_content")
                metadata = data.get("metadata", {})
            elif data.get("type") == "string" and "value" in data:
                page_content = data.get("value")
            else:
                # Unsupported format
                print(f"⚠️ Unsupported format in {file_path.name}; skipping")
                continue
        else:
            print(f"⚠️ Unsupported JSON root type in {file_path.name}; skipping")
            continue

        if not page_content:
            print(f"⚠️ No page content in {file_path.name}; skipping")
            continue

        # 1. Create the Embedding (Add 'passage: ' for better E5 accuracy)
        try:
            text_to_embed = f"passage: {page_content}"
            vector = embeddings_model.embed_query(text_to_embed)
        except Exception as e:
            print(f"❌ Embedding failed for {file_path.name}: {e}")
            continue

        # 2. Prepare for Supabase
        record = {
            "content": page_content,
            "metadata": metadata,
            "embedding": vector
        }

        # 3. Upload
        try:
            supabase.table("gita_contents").insert(record).execute()
            print(f"✅ Uploaded Page {metadata.get('source_page', file_path.stem)}")
        except Exception as e:
            print(f"❌ Error uploading {file_path.stem}: {e}")

if __name__ == "__main__":
    migrate_to_cloud()
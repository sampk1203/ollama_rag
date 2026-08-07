import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import SOURCE_DIRS, SUPPORTED_EXTENSIONS
from loaders import load_file_with_timeout


def get_indexed_files(vectorstore):
    indexed = set()
    batch_size = 1000
    offset = 0
    while True:
        try:
            result = vectorstore._collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"]
            )
            metadatas = result.get("metadatas", [])
            if not metadatas:
                break
            for m in metadatas:
                if m and "source" in m:
                    indexed.add(m["source"])
            if len(metadatas) < batch_size:
                break
            offset += batch_size
        except Exception as e:
            print(f"⚠ Error fetching indexed files at offset {offset}: {e}")
            break
    return indexed


def save_in_batches(vectorstore, docs, batch_size=50):
    """Save docs to vectorstore in safe batches."""
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        vectorstore.add_documents(batch)
        print(f"  💾 {i+len(batch)}/{len(docs)} chunks saved", flush=True)


def update_research_library(vectorstore, extra_files=None):
    print("  Checking indexed files...", flush=True)
    indexed_files = get_indexed_files(vectorstore)
    print(f"  {len(indexed_files)} files already indexed.")

    all_new_files = []
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            print(f"⚠ Directory not found, skipping: {source_dir}")
            continue
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, filename)
                    if file_path not in indexed_files:
                        all_new_files.append((source_dir, file_path))

    if extra_files:
        for fp in extra_files:
            if fp and fp not in indexed_files:
                all_new_files.append(("downloads", fp))

    total = len(all_new_files)
    if total == 0:
        print("✓ Database is already up to date.")
        return

    print(f"Found {total} new files to index.\n")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    skipped = 0

    for i, (source_dir, file_path) in enumerate(all_new_files, 1):
        rel_path = os.path.relpath(file_path, source_dir) if os.path.exists(source_dir) else file_path
        ext = os.path.splitext(file_path)[1].lower()
        print(f"[{i}/{total}] ({ext}) {rel_path}", flush=True)

        raw_docs = load_file_with_timeout(file_path, seconds=120)
        if raw_docs:
            chunks = splitter.split_documents(raw_docs)
            if chunks:
                print(f"  → {len(chunks)} chunks, saving...", flush=True)
                save_in_batches(vectorstore, chunks)
            else:
                vectorstore.add_documents([Document(
                    page_content="[SKIPPED — no extractable text]",
                    metadata={"source": file_path, "skipped": True}
                )])
                skipped += 1
        else:
            skipped += 1

        if i % 10 == 0 or i == total:
            print(f"\n  ── {i}/{total} files │ {skipped} skipped ──\n", flush=True)

    print(f"\n✓ Done. {total - skipped}/{total} files indexed, {skipped} skipped.")
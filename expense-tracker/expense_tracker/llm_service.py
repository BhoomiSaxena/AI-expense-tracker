import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from utils import is_income_category

try:
	import chromadb
	from sentence_transformers import SentenceTransformer
except ImportError as exc:
	raise ImportError(
		"Missing RAG dependencies. Install requirements.txt before running the app."
	) from exc

try:
	from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
	ChatGoogleGenerativeAI = None


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "expense_rag_context"

_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
	global _embedding_model
	if _embedding_model is None:
		_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
	return _embedding_model


def _read_kb_documents(knowledge_base_dir: str) -> List[Dict[str, str]]:
	base_path = Path(knowledge_base_dir)
	if not base_path.exists():
		return []

	docs: List[Dict[str, str]] = []
	for file_path in base_path.rglob("*"):
		if not file_path.is_file():
			continue

		suffix = file_path.suffix.lower()
		text = ""

		if suffix in {".md", ".txt"}:
			text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
		elif suffix == ".pdf":
			try:
				from pypdf import PdfReader

				reader = PdfReader(str(file_path))
				text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
			except Exception:
				text = ""

		if text:
			docs.append({"text": text, "source": str(file_path)})

	return docs


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
	if len(text) <= chunk_size:
		return [text]

	chunks = []
	start = 0
	while start < len(text):
		end = min(start + chunk_size, len(text))
		chunks.append(text[start:end])
		if end == len(text):
			break
		start = max(0, end - overlap)
	return chunks


def _transaction_context(df: pd.DataFrame) -> str:
	temp_df = df.copy()
	income_mask = is_income_category(temp_df["Category"])
	expense_df = temp_df[~income_mask].copy()
	expense_df["Spend"] = expense_df["Amount"].abs()

	if expense_df.empty:
		return "No expense transactions were found in the uploaded data."

	total_spend = float(expense_df["Spend"].sum())
	top_category = expense_df.groupby("Category")["Spend"].sum().sort_values(ascending=False).head(1)
	top_merchant = expense_df.groupby("Description")["Spend"].sum().sort_values(ascending=False).head(1)

	month_col = pd.to_datetime(temp_df["Date"], errors="coerce").dt.strftime("%Y-%m")
	monthly_spend = (
		expense_df.assign(Month=month_col)
		.groupby("Month")["Spend"]
		.sum()
		.sort_values(ascending=False)
		.head(3)
	)

	lines = [f"Total spend: {total_spend:.2f}"]
	if not top_category.empty:
		lines.append(f"Top spending category: {top_category.index[0]} ({float(top_category.iloc[0]):.2f})")
	if not top_merchant.empty:
		lines.append(f"Top merchant: {top_merchant.index[0]} ({float(top_merchant.iloc[0]):.2f})")

	if not monthly_spend.empty:
		lines.append("Highest spend months:")
		for month, value in monthly_spend.items():
			lines.append(f"- {month}: {float(value):.2f}")

	return "\n".join(lines)


def _transaction_documents(df: pd.DataFrame) -> List[Dict[str, str]]:
	rows = []
	for _, row in df.iterrows():
		rows.append(
			{
				"text": (
					f"Date: {row['Date']}; Description: {row['Description']}; "
					f"Amount: {row['Amount']}; Category: {row['Category']}"
				),
				"source": "uploaded_transactions",
			}
		)

	rows.append({"text": _transaction_context(df), "source": "transaction_summary"})
	return rows


def _get_collection(persist_directory: str):
	client = chromadb.PersistentClient(path=persist_directory)
	return client.get_or_create_collection(name=COLLECTION_NAME)


def build_vector_store(df: pd.DataFrame, knowledge_base_dir: str, persist_directory: str) -> None:
	client = chromadb.PersistentClient(path=persist_directory)
	try:
		client.delete_collection(name=COLLECTION_NAME)
	except Exception:
		pass

	collection = client.get_or_create_collection(name=COLLECTION_NAME)
	embedding_model = _get_embedding_model()

	base_docs = _transaction_documents(df) + _read_kb_documents(knowledge_base_dir)

	docs_to_add: List[str] = []
	ids_to_add: List[str] = []
	metas_to_add: List[Dict[str, str]] = []

	for doc_idx, doc in enumerate(base_docs):
		chunks = _chunk_text(doc["text"])
		for chunk_idx, chunk in enumerate(chunks):
			doc_id = f"{doc['source']}::{doc_idx}::{chunk_idx}"
			docs_to_add.append(chunk)
			ids_to_add.append(doc_id)
			metas_to_add.append({"source": doc["source"]})

	if docs_to_add:
		embeddings = embedding_model.encode(docs_to_add).tolist()
		collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add, embeddings=embeddings)


def retrieve_context(question: str, persist_directory: str, top_k: int = 4) -> List[Dict[str, str]]:
	collection = _get_collection(persist_directory)
	embedding_model = _get_embedding_model()
	query_vector = embedding_model.encode([question]).tolist()[0]

	results = collection.query(query_embeddings=[query_vector], n_results=top_k)
	documents = results.get("documents", [[]])[0]
	metadatas = results.get("metadatas", [[]])[0]

	context = []
	for idx, text in enumerate(documents):
		source = "unknown"
		if idx < len(metadatas) and isinstance(metadatas[idx], dict):
			source = metadatas[idx].get("source", "unknown")
		context.append({"text": text, "source": source})
	return context


def _llm_answer(question: str, context_chunks: List[Dict[str, str]]) -> str:
	api_key = os.getenv("GOOGLE_API_KEY")
	if not api_key or ChatGoogleGenerativeAI is None:
		return ""

	context_text = "\n\n".join(
		f"Source: {item['source']}\n{item['text']}" for item in context_chunks
	)

	prompt = f"""
You are a financial insights assistant.
Answer ONLY from the provided context.
If context is insufficient, clearly say what is missing.

User Question:
{question}

Context:
{context_text}

Return a concise, practical answer with 3-5 bullet points when possible.
"""

	llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
	response = llm.invoke(prompt)
	return (response.content or "").strip()


def _fallback_answer(question: str, context_chunks: List[Dict[str, str]]) -> str:
	summary_parts = []
	for item in context_chunks:
		if item["source"] in {"transaction_summary", "uploaded_transactions"}:
			summary_parts.append(item["text"])

	summary = "\n".join(summary_parts)[:1000]
	return (
		"Here is what I found from your uploaded context:\n\n"
		f"Question: {question}\n\n"
		f"Relevant data:\n{summary}\n\n"
		"Suggestions:\n"
		"- Track your highest spend category weekly.\n"
		"- Set a monthly cap for non-essential categories.\n"
		"- Move fixed costs and savings into separate budget buckets."
	)


def answer_finance_question(
	df: pd.DataFrame,
	question: str,
	knowledge_base_dir: str = "knowledge_base",
	persist_directory: str = ".chromadb",
) -> Dict[str, object]:
	build_vector_store(df=df, knowledge_base_dir=knowledge_base_dir, persist_directory=persist_directory)
	context = retrieve_context(question=question, persist_directory=persist_directory, top_k=4)

	llm_result = _llm_answer(question, context)
	if llm_result:
		return {"answer": llm_result, "context": context, "used_llm": True}

	return {"answer": _fallback_answer(question, context), "context": context, "used_llm": False}

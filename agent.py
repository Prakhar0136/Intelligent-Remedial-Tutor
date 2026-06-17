import os
import shutil
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================================================
# SCHEMAS
# ==================================================

class QuizQuestion(BaseModel):
    concept: str = Field(description="The core concept being tested.")
    question: str = Field(description="A multiple-choice question.")
    options: list[str] = Field(description="Exactly 4 answer choices.")
    correct_option: str = Field(description="Correct option letter A/B/C/D.")
    explanation: str = Field(description="Explanation of the correct answer.")

class EvaluationResult(BaseModel):
    is_correct: bool = Field(description="Whether student answer is correct.")
    diagnostic: str = Field(description="Reasoning behind correctness or misconception.")
    remediation_search_query: str = Field(description="Search query for remedial learning.")

# ==================================================
# AGENT
# ==================================================

class QuizzerAgent:
    def __init__(self, groq_api_key: str):
        self.persist_dir = "./chroma_db"
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=groq_api_key
        )
        self.quiz_llm = self.llm.with_structured_output(QuizQuestion)
        self.eval_llm = self.llm.with_structured_output(EvaluationResult)
        
        # Initialize empty vectorstore; will be populated by ingest_pdf
        self.vectorstore = None

    # ==================================================
    # PDF INGESTION
    # ==================================================
    def ingest_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Clear existing DB to prevent conflicts
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return len(chunks)

    # ==================================================
    # CORE AGENT LOGIC
    # ==================================================
    def generate_question(self, topic_query: str = "") -> QuizQuestion:
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Upload PDF first.")

        docs = self.vectorstore.similarity_search(topic_query or "introduction overview summary", k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert tutor. Create one 4-option MCQ from the reference material.\nReference Material:\n{context}"),
            ("human", "Generate the question.")
        ])
        return (prompt | self.quiz_llm).invoke({"context": context})

    def evaluate_answer(self, question_obj: QuizQuestion, student_answer: str) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Evaluate the student's answer. If incorrect, provide a diagnostic and a remediation search query."),
            ("human", "Concept: {concept}\nQuestion: {question}\nCorrect: {correct_option}\nStudent: {student_answer}")
        ])
        return (prompt | self.eval_llm).invoke({
            "concept": question_obj.concept,
            "question": question_obj.question,
            "correct_option": question_obj.correct_option,
            "student_answer": student_answer
        })

    def generate_remediation(self, eval_result: EvaluationResult) -> str:
        docs = self.vectorstore.similarity_search(eval_result.remediation_search_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly remedial tutor. Explain the concept simply using an analogy. Max 200 words."),
            ("human", "Diagnostic: {diagnostic}\nReference Material: {context}")
        ])
        return (prompt | self.llm).invoke({
            "diagnostic": eval_result.diagnostic,
            "context": context
        }).content
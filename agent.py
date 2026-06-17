import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load Groq API Key
load_dotenv()

# --- STEP 1: DEFINE SCHEMAS ---

class QuizQuestion(BaseModel):
    concept: str = Field(description="The core concept or topic being tested.")
    question: str = Field(description="A clear multiple-choice quiz question.")
    options: list[str] = Field(description="Exactly 1 multiple choice options (A, B, C, D).")
    correct_option: str = Field(description="The single letter corresponding to the right answer (A, B, C, or D).")
    explanation: str = Field(description="The correct explanation directly from the reference material.")

# New Schema for the Evaluator Chain
class EvaluationResult(BaseModel):
    is_correct: bool = Field(description="True if the student's answer is correct, False otherwise.")
    diagnostic: str = Field(description="If wrong, a deep analysis of the student's likely misconception (e.g., 'confused syntax with logic'). If correct, praise.")
    remediation_search_query: str = Field(description="A tailored search query targeting the underlying foundational concept to look up alternative explanations.")


# --- STEP 2: THE AGENT BRAIN ---

class QuizzerAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        
        # Structured LLMs
        self.quiz_llm = self.llm.with_structured_output(QuizQuestion)
        self.eval_llm = self.llm.with_structured_output(EvaluationResult)

    def generate_question(self, topic_query: str = "") -> QuizQuestion:
        """Fetches material and creates a structured quiz question."""
        if topic_query:
            docs = self.vectorstore.similarity_search(topic_query, k=2)
        else:
            docs = self.vectorstore.similarity_search("summary overview introduction", k=2)
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert tutor. Look at the Reference Material, extract a key concept, and create a 4-option multiple-choice question.\n\nReference Material:\n{context}"),
            ("human", "Generate a multiple choice quiz question.")
        ])
        
        chain = prompt | self.quiz_llm
        return chain.invoke({"context": context})

    def evaluate_answer(self, question_obj: QuizQuestion, student_answer: str) -> EvaluationResult:
        """Evaluates the user's answer and diagnoses the mistake if they are wrong."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert educational evaluator. Compare the student's answer with the correct answer. "
                "If the student is wrong, analyze the question and the correct explanation to determine their core misunderstanding. "
                "Then, create a specific search query to retrieve simpler foundational data or analogies from a documentation vector store."
            )),
            ("human", (
                "Concept being tested: {concept}\n"
                "Question: {question}\n"
                "Correct Option: {correct_option}\n"
                "Correct Explanation: {explanation}\n"
                "Student's Answer: {student_answer}\n\n"
                "Evaluate the student's response."
            ))
        ])
        
        chain = prompt | self.eval_llm
        return chain.invoke({
            "concept": question_obj.concept,
            "question": question_obj.question,
            "correct_option": question_obj.correct_option,
            "explanation": question_obj.explanation,
            "student_answer": student_answer
        })

    def generate_remediation(self, eval_result: EvaluationResult) -> str:
        """Uses the diagnostic search query to pull new info and explain it simply."""
        
        # 1. Dynamic fetch: Use the LLM's generated query to find a different part of the PDF
        print(f"[Agent Search Query] Searching DB for: '{eval_result.remediation_search_query}'")
        remediation_docs = self.vectorstore.similarity_search(eval_result.remediation_search_query, k=2)
        remediation_context = "\n\n".join([doc.page_content for doc in remediation_docs])
        
        # 2. Ask the LLM to rewrite this with a completely new approach/analogy
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a remedial tutor. The student failed a question due to this diagnostic: {diagnostic}.\n"
                "Use the following reference material to explain the concept in a completely different way, "
                "using a simple real-world analogy or a step-by-step breakdown to fix their specific confusion. Keep it encouraging but concise."
            )),
            ("human", "Reference Material:\n{context}\n\nProvide the breakdown and analogy:")
        ])
        
        # Standard string output since this is an explanation block for the student
        chain = prompt | self.llm
        response = chain.invoke({
            "diagnostic": eval_result.diagnostic,
            "context": remediation_context
        })
        
        return response.content


# --- STEP 3: LOCAL TERMINAL SIMULATION ---

if __name__ == "__main__":
    tutor = QuizzerAgent()
    print("Generating quiz question...")
    quiz = tutor.generate_question()
    
    print(f"\n[QUESTION]: {quiz.question}")
    for option in quiz.options:
        print(f"  {option}")
        
    # We will simulate a WRONG answer deliberately to test our Self-Correcting loop!
    fake_wrong_answer = "A" if quiz.correct_option != "A" else "B"
    print(f"\n[Simulated Student Input]: {fake_wrong_answer} (Correct answer was {quiz.correct_option})")
    
    print("\nEvaluating response...")
    evaluation = tutor.evaluate_answer(quiz, fake_wrong_answer)
    
    print(f"Is Correct? {evaluation.is_correct}")
    print(f"Diagnostic Analysis: {evaluation.diagnostic}")
    
    if not evaluation.is_correct:
        print("\n--- AGENT REMEDIATION STEP ---")
        explanation = tutor.generate_remediation(evaluation)
        print(explanation)
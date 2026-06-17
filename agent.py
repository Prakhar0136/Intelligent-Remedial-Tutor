import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load Groq API Key
load_dotenv()

# Define what a "Quiz Question" looks like using Pydantic
class QuizQuestion(BaseModel):
    concept: str = Field(description="The core concept or topic being tested.")
    question: str = Field(description="A clear multiple-choice quiz question.")
    options: list[str] = Field(description="Exactly 1 multiple choice options (A, B, C, D).")
    correct_option: str = Field(description="The single letter corresponding to the right answer (A, B, C, or D).")
    explanation: str = Field(description="The correct explanation directly from the reference material.")

class QuizzerAgent:
    def __init__(self):
        # 1. Connect to our existing Chroma DB
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        
        # 2. Initialize the blazing fast Groq LLM
        # We use llama-3.3-70b-versatile for high intelligence and reasoning capabilities
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        # 3. Bind the structured output to our LLM
        # This forces the LLM to output exactly according to our Pydantic schema
        self.structured_llm = self.llm.with_structured_output(QuizQuestion)

    def generate_question(self, topic_query: str = "") -> QuizQuestion:
        # Search the database for relevant reference content. 
        # If no specific query is given, we just pull a random chunk to start.
        if topic_query:
            docs = self.vectorstore.similarity_search(topic_query, k=2)
        else:
            # Fallback: get some content from the vector store to extract a concept from
            docs = self.vectorstore.similarity_search("summary overview introduction", k=2)
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt instructing the LLM to act as a strict teacher
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert tutor. Your task is to look at the provided Reference Material, "
                "extract an important technical concept, and create a single multiple-choice question to test the student.\n\n"
                "Reference Material:\n{context}"
            )),
            ("human", "Generate a challenging multiple choice quiz question based on the material.")
        ])
        
        # Combine the prompt and our structured LLM into a chain
        chain = prompt | self.structured_llm
        
        # Invoke the chain to generate the structured object
        response = chain.invoke({"context": context})
        return response

if __name__ == "__main__":
    # Quick testing script to see if it works!
    print("Initializing Quizzer Agent...")
    tutor = QuizzerAgent()
    
    print("Generating a quiz question based on your PDF...")
    quiz = tutor.generate_question()
    
    print("\n--- GENERATED QUIZ QUESTION ---")
    print(f"Concept: {quiz.concept}")
    print(f"Question: {quiz.question}")
    for idx, option in enumerate(quiz.options):
        print(f"  {option}")
    print(f"Correct Answer: {quiz.correct_option}")
    print(f"Reference Explanation: {quiz.explanation}")
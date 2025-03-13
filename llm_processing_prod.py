# Improved script for sequential question-answering with ChatOllama
# Based on test_llm_v3_scaled.py

import pandas as pd
import re
import json
import os
import time
import warnings
import traceback
import shutil
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple


# Suppress specific pandas warnings for CSV parsing
warnings.filterwarnings("ignore", message=".*error_bad_lines.*", category=FutureWarning)

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

# Enable LangSmith tracing
load_dotenv()

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2
# SAMPLE_SIZE = 1 # for testing
DEBUG = True

class QuestionParser:
    """Parse questions from the CSV file."""
    
    @staticmethod
    def parse_questions_from_csv(csv_path: str) -> List[Dict[str, Any]]:
        """
        Read questions from a CSV file where each row contains a question and examples.
        
        Returns a list of dictionaries, each containing:
        - question_text: The full text of the question including examples
        """
        questions = []
        
        # Read the CSV file
        try:
            # Read using pandas with appropriate parameters for multiline fields
            df = pd.read_csv(csv_path, engine='python')
            
            # Ensure required columns exist
            if "Question" not in df.columns:
                raise ValueError("CSV must contain a 'Question' column")
            
            # Process each question row
            for idx, row in df.iterrows():
                if pd.isna(row["Question"]):
                    continue  # Skip rows with empty questions
                    
                # Get the full question text
                question_text = str(row["Question"]).strip()
                
                # Get the main question (text before examples)
                main_question_pattern = r"^(.*?)(?=\s*-\s*Example\s*\d+:|$)"
                main_question_match = re.search(main_question_pattern, question_text, re.DOTALL)
                main_question = main_question_match.group(1).strip() if main_question_match else question_text
                
                # Create question data dictionary
                question_data = {
                    "question_num": idx + 1,
                    "question_text": main_question,
                    "full_content": question_text                }
                
                # Extract examples for this question using regex
                examples = []
                example_pattern = r"```json\s*([\s\S]*?)```"
                example_matches = re.findall(example_pattern, question_text, re.DOTALL)
                
                for example in example_matches:
                    examples.append(example.strip())
                
                question_data["examples"] = examples
                print(f"[Parser] Question #{idx+1}: '{main_question[:50]}...' with {len(examples)} examples")
                questions.append(question_data)
            
            if not questions:
                print("Warning: No valid questions were found in the CSV file.")
                
            return questions
            
        except Exception as e:
            print(f"Error parsing CSV file: {str(e)}")
            traceback.print_exc()
            return []

class MemoryManager:
    """Manage conversation memory for the complete set of questions for a document."""
    
    def __init__(self):
        self.memory = []  # No max limit - keep all interactions for this document
        self.all_responses = {}  # Store all responses for final compilation
        self.csv_answers = {}    # Store answers to be written back to CSV
        # print(f"[Memory] Initialized without item limit to retain all document interactions")
    
    def add_interaction(self, question: str, answer: Dict[str, Any], question_num: int = None) -> None:
        """Add a question-answer pair to memory."""
        # Store in full memory dict
        q_key = question.strip().split('\n')[0][:50]  # Use first line of question as key
        self.all_responses[q_key] = answer
        
        # Store for CSV output if question_num is provided
        if question_num is not None:
            self.csv_answers[question_num] = answer
        
        # Add to working memory (for context window) - no trim
        self.memory.append({"question": question, "answer": answer})
        
        # print(f"[Memory] Added: '{q_key}' | Memory size: {len(self.memory)} items")
    
    def get_context_window(self) -> str:
        """Get the current memory context as formatted text."""
        context = ""
        for item in self.memory:
            q = item["question"].strip().split('\n')[0][:100]  # Truncate for brevity
            a = json.dumps(item["answer"], indent=2)
            context += f"Q: {q}\nA: {a}\n\n"
        # print(f"[Memory] Built context window from {len(self.memory)} items | Size: {len(context)} chars")

        return context
    
    def get_csv_answers(self) -> Dict[int, Dict[str, Any]]:
        """Get all answers to be written to CSV."""
        return self.csv_answers

class RAGProcessor:
    """Process documents with Retrieval-Augmented Generation."""
    
    def __init__(self, model_name: str = "Llama3.1", csv_output_path: str = None):
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = self._init_llm()
        self.json_parser = JsonOutputParser()
        self.memory_manager = None  # Will be initialized per document
        self.csv_output_path = csv_output_path
        
        # Load the CSV for updating if provided
        if self.csv_output_path:
            try:
                self.output_df = pd.read_csv(self.csv_output_path, engine='python')
                print(f"[CSV] Loaded output CSV with {len(self.output_df)} rows")
            except Exception as e:
                print(f"[CSV] Error loading output CSV: {str(e)}")
                self.output_df = None
    
    def _init_llm(self) -> ChatOllama:
        """Initialize the LLM."""
        return ChatOllama(
            model=self.model_name,
            n_gpu_layers=-1,
            n_batch=512,
            num_ctx=512*18,
            max_tokens=2000,
            f16_kv=True,
            temperature=0,
            top_k=40,
            top_p=0.95,
            verbose=DEBUG
        )
    
    def _create_vector_store(self, documents: List) -> InMemoryVectorStore:
        """Create and populate a vector store from documents."""
        vector_store = InMemoryVectorStore(self.embeddings)
        vector_store.add_documents(documents=documents)
        return vector_store
    
    def update_csv_with_answer(self, question_num: int, answer: Dict[str, Any]) -> None:
        """Update the CSV file with the answer for a specific question."""
        if self.csv_output_path and self.output_df is not None:
            try:
                # Convert answer to JSON string
                answer_str = json.dumps(answer)
                
                # Update the specific row (question_num - 1 for 0-based indexing)
                if question_num - 1 < len(self.output_df):
                    self.output_df.at[question_num - 1, "LLM Answer"] = answer_str
                    
                    # Save the updated dataframe back to CSV
                    self.output_df.to_csv(self.csv_output_path, index=False)
                    print(f"[CSV] Updated question #{question_num} in {self.csv_output_path}")
                else:
                    print(f"[CSV] Warning: Question #{question_num} exceeds CSV row count ({len(self.output_df)})")
            except Exception as e:
                print(f"[CSV] Error updating CSV: {str(e)}")
    
    def process_document(self, text_data: str, questions: List[Dict[str, Any]]) -> None:
        """Process a single document with all questions sequentially."""
        # Create a fresh memory manager for this document
        self.memory_manager = MemoryManager()
        
        # Load and split document
        loader = DataFrameLoader(pd.DataFrame([{"Pre-processed Text": text_data}]), page_content_column="Pre-processed Text")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        # print(f"[Document] Split text ({len(text_data)} chars) into {len(all_splits)} chunks")
        
        # Process each question sequentially
        for question_data in questions:
            # Create a fresh vector store for each question to manage memory
            vector_store = self._create_vector_store(all_splits)
            
            # Process this individual question
            # print(f"\n[Processing] Question {question_data['question_num']}: '{question_data['question_text'][:50]}...'")
            response = self._process_question(
                vector_store, 
                text_data, 
                question_data["full_content"],
                question_num=question_data["question_num"]
            )
            
            # Add to memory
            self.memory_manager.add_interaction(
                question_data["question_text"], 
                response, 
                question_data["question_num"]
            )
            
            # Update the CSV with this answer
            self.update_csv_with_answer(question_data["question_num"], response)
            
            # Clean up to manage memory
            del vector_store
    
    def _process_question(self, vector_store: InMemoryVectorStore, 
                          text_data: str, question_text: str, 
                          step_info: str = None, question_num: int = None) -> Dict[str, Any]:
        """Process a single question against the document."""
        # System prompt with memory context
        system_prompt = f"""
        You are a helpful research assistant. Only respond in JSON.
        For JSON structure, follow the example(s) provided along with each question.
        
        {f"Here is the current step we are working on: {step_info}" if step_info else ""}
        
        Previous question-answer pairs:
        {self.memory_manager.get_context_window()}
        """
        
        # Retrieve relevant chunks for this specific question
        retrieved_docs = vector_store.similarity_search(question_text, k=3)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
        # print(f"[Retrieve] Found {len(retrieved_docs)} relevant chunks with {len(retrieved_text)} chars")
        
        # Format prompt for this question
        prompt_template = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template="""
            {system_prompt}
            
            Relevant context:
            {context}
            
            Current question:
            {question}
            """
        )
        
        formatted_prompt = prompt_template.format(
            system_prompt=system_prompt,
            context=retrieved_text,
            question=question_text
        )
        
        if DEBUG:
            # Show which question is being processed; cut off the question at the first question mark
            print(f"\n{'='*50}\nProcessing question #{question_num}: {question_text.split('?')[0]}?\n{'='*50}")
        
        # Try with retries
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Current prompt:\n{formatted_prompt}")
                response = self.llm.invoke(formatted_prompt)
                
                # Extract JSON from response
                response_text = response.content
                print(f"[LLM] LLM Response content:\n{response_text}")
                # print(f"[LLM] Response received for question #{question_num}, length: {len(response_text)} chars")
                
                # Try to parse the JSON
                try:
                    # Look for JSON block in the response using regex
                    json_pattern = r'```json\s*([\s\S]*?)\s*```'
                    json_match = re.search(json_pattern, response_text)
                    
                    if json_match:
                        json_str = json_match.group(1)
                        json_data = json.loads(json_str)
                        print(f"[JSON] Parsed from code block for question #{question_num}")
                    else:
                        # If not in code block, try parsing the whole text
                        json_data = json.loads(response_text)
                        print(f"[JSON] Parsed from whole text for question #{question_num}")
                    
                    return json_data
                    
                except json.JSONDecodeError:
                    if attempt < MAX_RETRIES - 1:
                        print(f"Failed to parse JSON (attempt {attempt+1}), retrying...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print("Failed to parse JSON after all retries. Returning text response.")
                        return {"text_response": response_text}
                        
            except Exception as e:
                if attempt < MAX_RETRIES - 1: # Invokes LLM again with the same prompt
                    print(f"Error: {str(e)}. Retrying ({attempt+1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                    return {"error": str(e)}
        
        return {"error": "Unknown error occurred"}

def create_csv_copy(input_csv, output_csv):
    """Create a copy of the template CSV file (master prompt with additional columns for LLM answers)."""
    try:
        # Make a copy of the original CSV file
        shutil.copy(input_csv, output_csv)
        print(f"[CSV] Created copy of {input_csv} as {output_csv}")
        return True
    except Exception as e:
        print(f"[CSV] Error creating CSV copy: {str(e)}")
        return False

def main():
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Source CSV file
    input_csv = "master_prompt_v5_cot_table.csv" ### THIS SHOULD BE INHERITED FROM ORCHESTRATOR.PY
    
    # Read in the preprocessed dataset
    df = pd.read_csv("Preprocessed/preprocessed_sources_20250313_162322.csv") ### THIS SHOULD BE INHERITED FROM ORCHESTRATOR.PY
    
    # Sample N rows 
    # df = df.sample(SAMPLE_SIZE)
    
    # Parse questions from the CSV file
    question_parser = QuestionParser()
    questions = question_parser.parse_questions_from_csv(input_csv)
    
    print(f"Loaded {len(questions)} questions from CSV file")
    if len(questions) == 0:
        print("No questions found in CSV.")
        return
        
    # Debug output to check the first few questions
    if DEBUG and questions:
        print("\nFirst question:")
        print(f"Text: {questions[0]['question_text'][:100]}...")
        print(f"Examples count: {len(questions[0]['examples'])}")
        if questions[0]['examples']:
            print(f"First example: {questions[0]['examples'][0][:100]}...")
    
    # Process each document with its own CSV output
    for index, row in df.iterrows():
        print(f"\n[Main] Processing document {index + 1}/{len(df)}...")
        
        # Grab the Event Group name to include in the output CSV filename
        if "Event Group" in row:
            event_group = row["Event Group"]
        elif "Event Group ID" in row:
            event_group = row["Event Group ID"]
        else:
            event_group = index + 1


        # Create a unique CSV output file for this document
        output_csv = f"Output/llm_answers_{timestamp}_{event_group}.csv" ### RETAIN THE FILENAME CONVENTION BUT LOCATION OF DIRECTORY SHOULD BE INHERITED FROM ORCHESTRATOR.PY
        
        # Create a copy of the master CSV file for this document
        if not create_csv_copy(input_csv, output_csv):
            print(f"[Main] Skipping document {index + 1} due to CSV creation error")
            continue
            
        # Create processor with document-specific CSV output path
        processor = RAGProcessor(csv_output_path=output_csv)
        
        text_data = row["Pre-processed Text"] # grab the text from the Pre-processed Text column
        print(f"[Document] Length: {len(text_data)} chars | Preview: {text_data[:100]}...")
        
        # Process all questions for this document - no return value needed
        processor.process_document(text_data, questions)
        
        print(f"[Main] CSV with answers saved to {output_csv}")
    
    print(f"[Main] Processing complete. Individual CSV files created for each document with prefix 'llm_answers_{timestamp}_'")

if __name__ == "__main__":
    main()
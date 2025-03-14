# Instances of ES RAG Processing Pipeline

This repository contains a pipeline for processing PDF documents associated with economic statecraft (ES) events using retrieval-augmented generation (RAG) to extract structured information and answer predefined questions.

## Overview

The pipeline processes PDF documents through several stages:
1. **Extraction & Preprocessing**: Extract text from PDFs and preprocess it for optimal RAG performance
2. **LLM Processing**: Prompt a local LLM with a series of questions with specific formatting examples
3. **Output Storage**: Each output from the LLM is stored in an event-specific CSV (i.e., each ES Event gets a CSV output)

## Repository Structure

```
.
├── Input/PDFs/           # Directory for input PDF files
├── Preprocessed/         # Directory for preprocessed data
├── Output/               # Directory for generated outputs
├── extract_preprocess_pdfs_prod.py  # PDF extraction and preprocessing script
├── llm_processing_prod.py           # LLM processing script for question answering
├── requirements.txt      # Python dependencies
```

## Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure you have [Ollama](https://ollama.ai/) installed and running with the required models:
   - Llama3.1
   - nomic-embed-text

## Usage

### 1. PDF Document Preparation

Place your PDFs in the `Input/PDFs/` directory following this naming convention:
```
EventGroup_Source_Number.pdf
```

For example:
- `ChinaEconomicStimulus_Reuters_01.pdf`
- `ChinaEconomicStimulus_Bloomberg_02.pdf`

### 2. Extract and Preprocess PDF Content

```bash
python extract_preprocess_pdfs_prod.py --input_folder Input/PDFs --output_csv Preprocessed/preprocessed_sources_[timestamp].csv
```

Optional arguments:
- `--input_folder`: Path to the folder containing PDFs (default: `Input/PDFs`)
- `--output_csv`: Path to save the CSV file (default: `Preprocessed/preprocessed_sources_[timestamp].csv`)

### 3. Run LLM Processing

```bash
python llm_processing_prod.py
```

This script:
- Reads the preprocessed CSV file
- Loads questions from the master prompt CSV
- Processes each document with all questions sequentially
- Generates output CSV files in the `Output/` directory for each Event Group (i.e., ES Event)

## Input Requirements

### PDF Naming Convention
Files must follow the format: `EventGroup_Source_Number.pdf`

Examples:
- `ChinaEconomicStimulus_Reuters_01.pdf`
- `UKBrexit_BBC_01.pdf`

### Master Prompt CSV
The LLM processing requires a master prompt CSV file (e.g., `master_prompt_v5_cot_table.csv`) with questions formatted as:
- A "Question" column containing the question text
- JSON examples in the format ```json {...} ``` for structured output

## Output

Output files are saved in the `Output/` directory as CSV files with the naming convention:
```
llm_answers_[timestamp]_[EventGroup].csv
```

Each CSV file contains:
- The original questions from the master prompt
- LLM-generated answers in JSON format

## Technical Details

### PDF Preprocessing

The preprocessing script:
- Extracts text and metadata from PDFs
- Cleans and normalizes text (removes artifacts, fixes encoding issues) using regex
- Groups PDFs by event for consolidated analysis

### LLM Processing

The LLM processing script:
- Uses RAG with in-memory vector stores for relevant context retrieval
- Processes questions sequentially, building context from previous answers
- Handles JSON parsing and formatting for structured outputs

## Dependencies

Key dependencies include:
- langchain and langchain-community for RAG components
- pdfplumber for PDF text extraction
- pandas for data handling
- ollama for LLM processing

See `requirements.txt` for a complete list.

## Notes

- The system is designed to process PDFs in batches organized by event groups
- For large documents, consider adjusting the chunk size and overlap in the RAG processor
- JSON output format follows the examples provided in the master prompt
- JSON output can be erroneous
- LLM-generated answers can be erroneous - Human evaluation is required

## License

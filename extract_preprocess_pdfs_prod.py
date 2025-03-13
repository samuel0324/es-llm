import os
import re
import json
import hashlib
import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader
import argparse
from datetime import datetime


# Text preprocessing utilities
def preprocess_text_for_rag(text):
    """
    Comprehensive text preprocessing for RAG applications.
    Cleans PDF extraction artifacts and improves text quality for better retrieval.
    
    Args:
        text (str): Raw text extracted from PDFs
        
    Returns:
        str: Preprocessed text optimized for RAG
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Fix structural issues
    
    # Fix hyphenated words at line breaks (common PDF extraction issue)
    processed = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Step 2: Remove or normalize unnecessary elements
    
    # Replace URLs with [URL] placeholder to retain context
    processed = re.sub(r'https?://[^\s]+', '[URL]', processed)
    
    # Replace email addresses with [EMAIL] placeholder
    processed = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', processed)
    
    # Step 3: Normalize text formatting
    
    # Normalize quotes and special characters
    processed = processed.replace('â€œ', '"').replace('â€', '"')  # Common encoding issues
    processed = processed.replace('â€™', "'").replace('â€˜', "'")
    processed = processed.replace('\u2018', "'").replace('\u2019', "'")  # Smart quotes
    processed = processed.replace('\u201c', '"').replace('\u201d', '"')
    processed = processed.replace('\u2013', '-').replace('\u2014', '--')  # En/em dashes
    
    # Step 4: Clean up PDF extraction artifacts
    
    # Remove standalone page numbers
    processed = re.sub(r'\n\s*\d+\s*\n', '\n', processed)
    
    # Remove header/footer patterns
    processed = re.sub(r'Page \d+ of \d+', '', processed)
    
    # Remove common PDF artifacts like strange characters
    processed = re.sub(r'[\u0080-\u00FF]', '', processed)
    
    # Step 5: Improve text structure
    
    # Fix empty titles
    processed = re.sub(r'Title:\s*\n', 'Title: [No title provided]\n', processed)
    
    # Normalize whitespace
    processed = re.sub(r' {2,}', ' ', processed)  # Multiple spaces to single space
    processed = re.sub(r'\n{3,}', '\n\n', processed)  # Multiple newlines to double newline
    
    # Step 6: Format for better RAG chunking
    
    # Clearly separate articles if multiple exist in the text
    processed = re.sub(r'(\nTitle:|\nSource:)', r'\n\n---\n\1', processed)
    
    # Format potential heading sections
    processed = re.sub(r'\n([A-Z][A-Z\s]{5,})\n', r'\n\n## \1\n\n', processed)
    
    # Step 7: Final cleanup
    
    # Remove consecutive duplicate lines (common in PDFs)
    lines = processed.split('\n')
    unique_lines = []
    for i, line in enumerate(lines):
        if i == 0 or line.strip() != lines[i-1].strip():
            unique_lines.append(line)
    processed = '\n'.join(unique_lines)
    
    # Remove redundant whitespace at the end
    processed = processed.strip()
    
    return processed


# Return a list of dicts containing Event Group and Source pairs 
# ex. [{'Event Group': 'ChinaEconomicStimulus', 'Source': 'Reuters'}]
def parse_filenames(folder_path):
    '''
    Parses filenames in the folder to extract Event Group and Source pairs.

    01-07-2025 note: check if the input filename matches the naming format. If not, raise error?

    '''
    
    pattern = r"^(.*?)_(.*?)_\d+\.pdf$"

    event_group_source_pair = []
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            event_group = match.group(1)
            source = match.group(2)
            event_group_source_pair.append({"Event Group": event_group, "Source": source})
    
    return event_group_source_pair    

# Group PDFs by prefixes of the file names
def group_pdfs_by_event(folder_path):
    '''
    Groups PDFs by their EventGroup.
    ''' 
    pattern = r"^(.*?)_(.*?)_\d+\.pdf$"

    pdf_groups = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            match = re.match(pattern, filename)
            if match:
                event_group = match.group(1)
                if event_group not in pdf_groups:
                    pdf_groups[event_group] = []
                pdf_groups[event_group].append(filename)
    return pdf_groups



# Return article title and text in a dict
def extract_text_and_metadata(pdf_path):
    '''
    Extracts text and title metadata from a PDF using PDFPLumberLoader
    '''
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    metadata = docs[0].metadata # dict
    
    # Check if the metadata contains the article title
    if 'Title' in metadata.keys():
        title = docs[0].metadata['Title']
    else:
        title = ""

    # Combine all text from the document into a single string
    text = ' '.join([doc.page_content for doc in docs])
    return {'Title': title, 'Text': text}

def create_dataframe(folder_path):
    '''
    Creates a pandas DataFrame with grouped PDF information and merged text.
    Also includes preprocessed text optimized for RAG.
    '''

    grouped_pdfs = group_pdfs_by_event(folder_path)
    
    data = []

    for event_group, filenames in grouped_pdfs.items():
        # Generate a unique ID for the event group
        event_group_id = hashlib.md5(event_group.encode()).hexdigest()
        merged_text = []
        sources = []

        for filename in filenames:
            source = filename.split('_')[1]
            sources.append(source)
            
            pdf_path = os.path.join(folder_path, filename)

            # Extract title and text from PDF
            extracted = extract_text_and_metadata(pdf_path)

            merged_text.append(f"Title: {extracted['Title']}\nSource: {source}\n{extracted['Text']}\n")

        merged_text = '\n'.join(merged_text)
        
        # Apply text preprocessing for RAG
        preprocessed_text = preprocess_text_for_rag(merged_text)

        data.append({
            'Event Group': event_group,
            'Event Group ID': event_group_id,
            'Filenames in Event Group': filenames,
            'Sources in Event Group': sources,
            'Text': merged_text,
            'Pre-processed Text': preprocessed_text,
            'Preprocessing Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    df = pd.DataFrame(data)
    return df

def append_to_csv(df, csv_path):
    """
    Appends the DataFrame to an existing CSV or creates a new one if it doesn't exist.
    Handles the case where the CSV structure might differ.
    """
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        
        # Check if the existing CSV has the 'Pre-processed Text' column
        if 'Pre-processed Text' not in existing_df.columns and 'Pre-processed Text' in df.columns:
            # Add the column to the existing dataframe
            existing_df['Pre-processed Text'] = None
            existing_df['Preprocessing Timestamp'] = None
            
        # Ensure both dataframes have the same columns before concatenating
        for col in df.columns:
            if col not in existing_df.columns:
                existing_df[col] = None
                
        for col in existing_df.columns:
            if col not in df.columns:
                df[col] = None
                
        # Reorder columns to match
        df = df[existing_df.columns]
        
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
        print(f"DataFrame appended to {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV file at {csv_path}")

if __name__ == '__main__':
    # Set default paths
    default_input_folder = "Input/PDFs" ### THIS SHOULD BE INHERITED FROM ORCHESTRATOR.PY
    curr_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    default_output_csv = f"Preprocessed/preprocessed_sources_{curr_timestamp}.csv" ### THIS SHOULD BE INHERITED FROM ORCHESTRATOR.PY
    
    parser = argparse.ArgumentParser(description="Group PDFs, extract text, and preprocess for RAG.")
    
    # All arguments are optional with sensible defaults
    parser.add_argument("--input_folder", type=str, default=default_input_folder,
                        help=f"Path to the folder containing PDFs. Default: '{default_input_folder}'")
    
    parser.add_argument("--output_csv", type=str, default=default_output_csv,
                        help=f"Path to save the CSV file. Default: '{default_output_csv}'")
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Warning: Input folder '{args.input_folder}' does not exist. Please create it and add PDF files.")
        print(f"Files should follow the naming pattern: EventGroup_Source_Number.pdf")
        print(f"Example: ChinaEconomicStimulus_Reuters_01.pdf")
        exit(1)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Process PDFs and create dataframe
    print(f"Processing PDFs from: {args.input_folder}")
    print(f"Output will be saved to: {args.output_csv}")
    
    output_df = create_dataframe(args.input_folder)
    append_to_csv(output_df, args.output_csv)
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print(f"Processed {len(output_df)} event groups")
    
    orig_len = output_df['Text'].str.len().sum()
    proc_len = output_df['Pre-processed Text'].str.len().sum()
    reduction = (1 - proc_len / orig_len) * 100 if orig_len > 0 else 0
    
    print(f"Original text total length: {orig_len:,} characters")
    print(f"Preprocessed text total length: {proc_len:,} characters")
    print(f"Size reduction: {reduction:.2f}%")

# Example usage in terminal:
#
# 1. Basic usage (uses default paths):
#    python enhanced_process_pdfs.py
#
# 2. Specify a custom input folder:
#    python enhanced_process_pdfs.py --input_folder /path/to/pdf/folder
#
# 3. Specify a custom output CSV path:
#    python enhanced_process_pdfs.py --output_csv my_output_folder/processed_data.csv
#
# 4. Specify both custom input and output paths:
#    python enhanced_process_pdfs.py --input_folder /path/to/pdf/folder --output_csv my_output_folder/processed_data.csv
#
# Note:
# - PDF files must follow the naming convention: EventGroup_Source_Number.pdf
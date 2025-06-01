import pandas as pd
import json
import os

def convert_jsonl_to_csv(jsonl_file_path, csv_file_path):
    """
    Converts a JSONL file with a specific nested structure (from perspective.py) to a CSV file.
    """
    all_records = []
    
    # Define expected score keys to ensure all columns are present even if some scores are missing
    # Based on the 'requestedAttributes' in perspective.py
    expected_score_keys = [
        'TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 
        'PROFANITY', 'THREAT', 'SEXUALLY_EXPLICIT', 'FLIRTATION'
    ]

    with open(jsonl_file_path, 'r') as f_jsonl:
        for line in f_jsonl:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                continue

            # Extract base attributes, excluding 'scores'
            base_attributes = {k: v for k, v in data.items() if k != 'scores'}
            
            scores_data = data.get('scores', {})
            if not scores_data:
                # Handle cases where 'scores' might be missing or empty for a record
                # Create a record with base attributes and NaN for scores
                record = base_attributes.copy()
                record['prompt'] = None # Or some placeholder
                for score_key in expected_score_keys:
                    record[score_key] = None
                all_records.append(record)
                continue

            for prompt_name, prompt_scores in scores_data.items():
                record = base_attributes.copy()
                record['prompt'] = prompt_name
                
                attribute_scores = prompt_scores.get('attributeScores', {})
                
                for score_key in expected_score_keys:
                    score_detail = attribute_scores.get(score_key, {})
                    summary_score = score_detail.get('summaryScore', {})
                    record[score_key] = summary_score.get('value')

                all_records.append(record)

    if not all_records:
        print("No records found or processed. CSV file will be empty or not created.")
        # Optionally create an empty CSV with headers if desired
        # df = pd.DataFrame(columns=[...list of expected columns...])
        # df.to_csv(csv_file_path, index=False)
        return

    df = pd.DataFrame(all_records)
    
    # Reorder columns for better readability: base attributes first, then prompt, then scores
    # Get base attribute columns from the first record (if available)
    if all_records:
        first_record_base_cols = [k for k in all_records[0].keys() if k not in ['prompt'] + expected_score_keys]
        ordered_columns = first_record_base_cols + ['prompt'] + expected_score_keys
        # Ensure all columns in df are included, in case some records had extra keys
        # and to prevent KeyError if a column is not in ordered_columns
        final_columns = [col for col in ordered_columns if col in df.columns]
        # Add any columns present in df but not in ordered_columns (e.g. unexpected keys)
        for col in df.columns:
            if col not in final_columns:
                final_columns.append(col)
        df = df[final_columns]

    df.to_csv(csv_file_path, index=False)
    print(f"Successfully converted '{jsonl_file_path}' to '{csv_file_path}'")

if __name__ == '__main__':
    # Define the input and output file paths
    # Assuming the script is in multimodal_cognitive_ai/Uncovering_LVLM_Bias/
    # and the data is in a subdirectory outputs_vllm/physical_gender/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_jsonl_file = os.path.join(script_dir, 'outputs_vllm', 'physical_gender', 'perspective_api_all_occupations_all_gens_2_prompts_0.jsonl')
    output_csv_file = os.path.join(script_dir, 'outputs_vllm', 'physical_gender', 'perspective_api_all_occupations_all_gens_2_prompts_0.csv')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    
    convert_jsonl_to_csv(input_jsonl_file, output_csv_file)

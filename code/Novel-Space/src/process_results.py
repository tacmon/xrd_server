import os
import sys

# Set working directory to project root for easy path access
# Get the absolute path of the directory containing this script (src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Novel-Space/)
base_dir = os.path.dirname(script_dir)
# Change the current working directory to Novel-Space/
os.chdir(base_dir)
# Add the project root to sys.path so autoXRD can be imported
root_dir = os.path.dirname(base_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import pandas as pd
import numpy as np
import ast
import re

def parse_list_string(s):
    """
    Parses a string representation of a list, handling 'np.float64(...)' and other quirks.
    """
    if pd.isna(s) or s == "":
        return []
    # Replace np.float64(value) with just value, handling potential spaces
    s_clean = re.sub(r'np\.float64\s*\(\s*(.*?)\s*\)', r'\1', s)
    try:
        # ast.literal_eval handles standard list/string/number representations
        return ast.literal_eval(s_clean)
    except (ValueError, SyntaxError):
        # Fallback for more complex cases if needed
        return []

def process_results(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    df = pd.read_csv(input_file)
    
    # 1. Identify all unique tags
    all_tags = set()
    parsed_phases = []
    parsed_confidences = []
    
    for idx, row in df.iterrows():
        phases = parse_list_string(row['Predicted phases'])
        confidences = parse_list_string(row['Confidence'])
        
        parsed_phases.append(phases)
        parsed_confidences.append(confidences)
        
        for p in phases:
            all_tags.add(p)
    
    if not all_tags:
        print("No tags found in the 'Predicted phases' column.")
        return
    
    # 2. Let the user pick multiple tags
    sorted_tags = sorted(list(all_tags))
    print("\nAvailable tags in the CSV:")
    for i, tag in enumerate(sorted_tags, 1):
        print(f"{i}. {tag}")
    
    while True:
        try:
            choice_str = input("\nPlease select the numbers of the 'Main Substances' (主要物质, separated by commas or spaces): ")
            # Split by comma or space and convert to indices
            selections = [int(x.strip()) - 1 for x in re.split(r'[,\s]+', choice_str) if x.strip()]
            
            if all(0 <= idx < len(sorted_tags) for idx in selections) and selections:
                main_substances = set(sorted_tags[idx] for idx in selections)
                break
            else:
                print(f"Invalid choices. Please enter numbers between 1 and {len(sorted_tags)}.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
    
    selected_list = sorted(list(main_substances))
    print(f"\nProcessing with Main Substances: {selected_list}")
    
    # 3. Process the data
    processed_data = []
    for i in range(len(df)):
        phases = parsed_phases[i]
        confidences = parsed_confidences[i]
        
        # Default as Unidentified
        final_phase = "未识别"
        final_confidence = ""
        
        if phases:
            # Find the substance with the absolute highest confidence in this row
            max_conf = -1
            max_phase = None
            
            for p, c in zip(phases, confidences):
                if c > max_conf:
                    max_conf = c
                    max_phase = p
            
            # Check if this "winner" is one of the main substances and > 50%
            if max_phase in main_substances and max_conf > 50:
                final_phase = max_phase
                final_confidence = max_conf
        
        processed_data.append({
            'Filename': df.iloc[i]['Filename'],
            'Predicted phases': final_phase,
            'Confidence': final_confidence
        })
    
    output_df = pd.DataFrame(processed_data)
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to '{output_file}'.")

if __name__ == "__main__":
    process_results("result.csv", "processed_result.csv")

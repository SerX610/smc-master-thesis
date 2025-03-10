import pandas as pd
import json
import os

def main():
    # Load the annotations file
    annotations_file = "../../data/mtt/annotations.csv"
    annotations_df = pd.read_csv(annotations_file, delimiter='\t')
    
    # Extract tag names (skip the first and last columns)
    tag_names = annotations_df.columns[1:-1].tolist()
    
    # Create a dictionary mapping index to tag
    tag_index_mapping = {index: tag for index, tag in enumerate(tag_names)}
    
    # Define the output JSON file path
    output_json_file = "../../data/mtt/tag_index_mapping.json"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    
    # Save the mapping to a JSON file
    with open(output_json_file, 'w') as f:
        json.dump(tag_index_mapping, f, indent=4)
    
    print(f"Tag index mapping saved to {output_json_file}")

if __name__ == "__main__":
    main()
import sys
import os
import json

def run():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # CodaBench V2 logic: Look for ingestion output in 'res'
    score_file = os.path.join(input_dir, 'res', 'scores.json')
    if not os.path.exists(score_file):
        score_file = os.path.join(output_dir, 'scores.json')

    score = 0
    if os.path.exists(score_file):
        with open(score_file, 'r') as f:
            data = json.load(f)
            score = data.get('score', 0)
        
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({"score": score}, f)

if __name__ == "__main__":
    run()
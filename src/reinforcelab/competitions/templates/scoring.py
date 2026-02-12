import os
import json

def run():
    # Codabench uses fixed paths under /app (no argv)
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # CodaBench V2 logic: Look for ingestion output in 'res'
    score_file = os.path.join(input_dir, 'res', 'scores.json')
    if not os.path.exists(score_file):
        score_file = os.path.join(output_dir, 'scores.json')

    data = {}
    if os.path.exists(score_file):
        with open(score_file, 'r') as f:
            data = json.load(f)
    # Pass through all keys; default score for eval-only phase
    if not data:
        data = {"score": 0}
        
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    run()
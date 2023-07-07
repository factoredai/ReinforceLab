import os
import pandas as pd
from datetime import datetime
import sys

def load_leaderboard(path='/leaderboard/leaderboard.csv'):

    path = "/".join((sys.argv[0]).split('/')[:-2]) + path

    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=['User', 'Score', "Num Epochs", 'Date (UTC)'])

def add_score(leaderboard, user="test_user", score=0, num_epochs=0):
    
    new_score = pd.DataFrame.from_dict({'User': [user], 'Score': [score], "Num Epochs": [num_epochs], 'Date (UTC)': [datetime.utcnow()]})
    leaderboard = pd.concat([leaderboard, new_score], axis=0)

    # Sort leaderboard
    leaderboard = leaderboard.sort_values(by=['Score', 'Num Epochs'], ascending=[0, 1]).reset_index(drop=True)

    return leaderboard

def save_leaderboard(leaderboard, path='/leaderboard/leaderboard.csv'):
    path = "/".join((sys.argv[0]).split('/')[:-2]) + path
    directory = "/".join(path.split('/')[:-1])
    if not os.path.isdir(directory):
        os.mkdir(directory)

    leaderboard.to_csv(path, index=False)

def update_readme(leaderboard):
    path = "/".join((sys.argv[0]).split('/')[:-2])
    f = open(f"{path}/README.md", "w")
    f.write("# Leaderboard \n\n")
    f.write(f"Last Update (UTC): {datetime.utcnow()}\n\n")
    f.write(leaderboard.to_markdown())
    f.write("\n")
    f.close()

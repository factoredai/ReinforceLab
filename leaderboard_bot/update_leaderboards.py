import os
import sys
from datetime import datetime

print(sys.argv[1:])

if len(sys.argv[1:-1]) > 0:
    for agent_path in sys.argv[1:-1]:

        environment_path = "/".join(agent_path.split("/")[:-1])
        os.system(f'python -m pip install -r {environment_path}/requirements.txt')
        os.system(f'python {environment_path}/update_leaderboard.py {sys.argv[-1]}')

f = open(f"leaderboard_bot/last_update.txt", "w")
f.write(f"Last Update (UTC): {datetime.utcnow()}\n\n")
f.close()

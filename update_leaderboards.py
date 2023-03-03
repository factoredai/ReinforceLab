import os
import sys
print(sys.argv[1:])

for agent_path in sys.argv[1:-1]:

    environment_path = "/".join(agent_path.split("/")[:-1])
    os.system(f'python {environment_path}/update_leaderboard.py {sys.argv[-1]}')

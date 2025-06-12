import subprocess
from sumolib import checkBinary
import time
import traci
import re

sumoBinary = "sumo"
sumoCmd = [sumoBinary, "-c", "in/osm.sumocfg"]
traci.start(sumoCmd)

NODES = {}

ids = traci.trafficlight.getIDList()
# print(ids)

# print(len(traci.trafficlight.getAllProgramLogics('254581852')[0].getPhases()))
ids = [id for id in ids if len(traci.trafficlight.getAllProgramLogics(id)[0].getPhases()) > 2]

PHASES = {}
temp_phases = []
for id in ids:
    phases = traci.trafficlight.getAllProgramLogics(id)[0].getPhases()
    phase_list = []
    for phase in phases:
        if 'y' not in phase.state:
            phase_list.append(phase.state)
    temp_phases.append(phase_list)
    
phases_set = []
for t in temp_phases:
    if t not in phases_set:
        phases_set.append(t)
        l = float(len(t))
        while str(round(l, 1)) in PHASES:
            l += 0.1
        PHASES[str(round(l, 1))] = t
        
# print(PHASES)

for id, t in zip(ids, temp_phases):
    for phase_id, phase in PHASES.items():
        if t == phase:
            NODES[id] = (phase_id, [])
            break


print(NODES)

with open('phases.txt', 'w') as f:
    phases_data = re.sub("\], ", "],\n", str(PHASES))
    f.write(phases_data)
    
with open('nodes.txt', 'w') as f:
    nodes_data = re.sub("\), ", "),\n", str(NODES))
    f.write(nodes_data)



# print(ids)
# print(len(ids))
traci.close()

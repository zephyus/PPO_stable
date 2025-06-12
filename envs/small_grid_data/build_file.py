import configparser
import logging
import numpy as np
import os

ILD_POS = 50

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
        
def output_flows(flow_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    flows = []
    flows.append(('-h11', '-h13'))
    flows.append(('-h21', '-h23'))
    flows.append(('-v11', '-v13'))
    flows.append(('-v21', '-v23'))
    # <routes>
    # <flow id="flow_1" from="-h11" to="-h13" begin="0" end="100000" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    # <flow id="flow_2" from="-h21" to="-h23" begin="0" end="100000" probability="0.1" departSpeed="max" departPos="base" departLane="best"/>
    # <flow id="flow_3" from="-v11" to="-v13" begin="0" end="100000" probability="0.1" departSpeed="max" departPos="base" departLane="best"/> 
    # <flow id="flow_4" from="-v21" to="-v23" begin="0" end="100000" probability="0.1" departSpeed="max" departPos="base" departLane="best"/> 
    # </routes>

    flow_str = '  <flow id="f%s" departPos="base" from="%s" to="%s" begin="%d" end="%d" probability="0.1" departSpeed="max" departLane="best" type="car"/>\n'
    output = '<routes>\n'
    output += '  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1" minGap="6"/>\n'
    
    t_begin, t_end = 0, 3600
    for i, flow in enumerate(flows):
        src, sink = flow
        output += flow_str % ('flow_'+str(i+1), src, sink, t_begin, t_end)
    output += '</routes>\n'
    return output

def output_config(thread=None):
    if thread is None:
        out_file = 'most.rou.xml'
    else:
        out_file = 'most_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="in/2x2.net.xml"/>\n'
    str_config += '    <route-files value="in/%s"/>\n' % out_file
    # str_config += '    <additional-files value="in/most.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config

def gen_rou_file(path, flow_rate, seed=None, thread=None):
    if thread is None:
        flow_file = 'most.rou.xml'
    else:
        flow_file = 'most_%d.rou.xml' % int(thread)
    write_file(path + 'in/' + flow_file, output_flows(flow_rate, seed=seed))
    sumocfg_file = path + ('most_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file
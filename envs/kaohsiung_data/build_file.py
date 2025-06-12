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

    flow_rates = [[250, 250, 290, 290, 290, 290, 185, 185, 295, 295],
                 [270,270, 305, 305, 305, 305, 180, 180, 340, 340],
                 [210,210, 245, 245, 245, 245, 155, 155, 320, 320]
                 ] # 0 morning peek # 1 afternoon peek # 2 off peek
    times = [[0, 600], [601, 3000], [3001, 3600]]
    FLOW_NUM = 6
    flows = []

    flows.append(('265614897#0', '92071741#6', '403863070#0 -272200697#3 -272200697#2 -272200697#1'))    # 大中-至真 西南向-1
    flows.append(('265614897#0', '-184383558#7', '383026340#0')) # 大中-至真 西南向-2
    flows.append(('403863070#1', '51005979#3', '383026340#0'))    # 至真-大順 西南向-1
    flows.append(('257245112#0', '-184383555', '383026340#0'))    # 至真-大順 西南向-2
    flows.append(('-401959564#4', '255537687#1', '383026340#4'))  # 至真-大順 西南向-3
    flows.append(('-88867314#4', '35164001#0', '383026340#1'))    # 至真-大順 西南向-4
    
    flows.append(('-51005979#3', '-265614897#0', '383026348#0'))  # 大順-至真-大中 東西向-1
    flows.append(('296215789#0', '-265614897#1', '23506825 383026348#0 383026348#1 383026348#3'))  # 大順-至真-大中 東西向-2
    flows.append(('270585766#0', '330465829#3', '23506825 383026348#0 383026348#1 330465827#2'))   # 大順-至真 東西向-3
    flows.append(('184383555', '401959564#4', '255537686#1'))     # 大順-至真 東西向-4

    flow_str = '  <flow id="f%s" departPos="random_free" from="%s" to="%s" via="%s" begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
    output = '<routes>\n'
    output += '  <vType id="car" length="4" accel="5" decel="10" speedDev="0.1" minGap="8"/>\n'
    
    for time, flow_rate in zip(times, flow_rates):
        t_begin, t_end = time
        for i, flow in enumerate(flows):
            src, sink, via = flow
            output += flow_str % (str(t_begin)+str(i), src, sink, via, t_begin, t_end, flow_rate[i])
    output += '</routes>\n'
    return output

def output_config(thread=None):
    if thread is None:
        out_file = 'most.rou.xml'
    else:
        out_file = 'most_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="in/most.net.xml"/>\n' #Mar19 from in/kaohsiung.net.xml to in/most.net.xml since kaohsiung is not found
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
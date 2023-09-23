import numpy as np
import ios

def spp_network_2():
    
    #Convolution Layer
    v = ios.placeholder(output_shape=(3, 100, 100))
    block = ios.Block(enter_node=v.node)
    v1 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(5, 5), stride=(1, 1), padding=(1, 1), act='relu')
    v1 = ios.pool2d(block, inputs=[[v1]], pool_type = 'max', kernel=(2, 2), stride=(2, 2))
    v1 = ios.conv2d(block, inputs=[[v1]], out_channels=128, kernel=(5, 5), stride=(1, 1), padding=(3, 3), act='relu')
    v1 = ios.pool2d(block, inputs=[[v1]], pool_type = 'max', kernel=(2, 2), stride=(2, 2))
    v1 = ios.conv2d(block, inputs=[[v1]], out_channels=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v1 = ios.pool2d(block, inputs=[[v1]], pool_type = 'max', kernel=(2, 2), stride=(2, 2))
    out = ios.identity(block, inputs=[[v1]], is_exit=True)  # concat v1, v2, and v3
    
    #SPP Layer
    adaptive_layer = ios.Block(enter_node=out.node)
    v1 = ios.pool2d(adaptive_layer, inputs=[[out]], pool_type='global_max')
    v2 = ios.pool2d(adaptive_layer, inputs=[[out]], pool_type='global_max')
    v3 = ios.pool2d(adaptive_layer, inputs=[[out]], pool_type='global_max')
    out = ios.identity(adaptive_layer, inputs=[[v1], [v2], [v3]], is_exit=True)

    graph = ios.Graph(name="DFG", input=v.node, blocks=[block, adaptive_layer])
    graph.init_weights()
    return graph

# model graph
graph = spp_network_2()

# measure latency
graph.sequential_schedule()
seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=16, repeat=6, profile_stage=True)
print(graph)
print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')

# optimize execution schedule
optimized_graph = ios.optimize(graph, batch_size=1, opt_type='dp_parallel', compute_weight=True)

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6, profile_stage=True)
print(optimized_graph)
print(f'Optimized schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}')

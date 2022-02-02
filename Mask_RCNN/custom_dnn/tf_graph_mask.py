import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

# TODO: point to where the graph is located
GRAPH_PB_PATH = 'frozen_inference_graph_coco.pb'

#print(GRAPH_PB_PATH)

#print(tf.compat.v1.Session())


with tf.compat.v1.Session() as sess:
   #print("load graph")

   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.compat.v1.GraphDef()
       #print(graph_def)
       graph_def.ParseFromString(f.read())
       sess.graph.as_default()
       tf.import_graph_def(graph_def, name='')
       graph_nodes = [n for n in graph_def.node]
       #print(graph_nodes)
       names = []
       for t in graph_nodes:
           #print(t.name)
           names.append(t.name)

print("------------Name List------------")
print(names)

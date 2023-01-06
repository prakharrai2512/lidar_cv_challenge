import numpy as np

import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate

COLOR_MAP = np.array(['#f59664', '#f5e664', '#963c1e', '#b41e50', '#ff0000',
                      '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff', '#ff96ff',
                      '#4b004b', '#4b00af', '#00c8ff', '#3278ff', '#00af00',
                      '#003c87', '#50f096', '#96f0ff', '#0000ff', '#ffffff'])

LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                      19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                      19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                      19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

# Load the LiDAR data and its label
lidar = np.fromfile('assets/000004.bin', dtype=np.float32)
label = np.fromfile('assets/000004.label', dtype=np.int32)
lidar = lidar.reshape(-1, 4)
label = LABEL_MAP[label & 0xFFFF]

# Filter out ignored points
lidar = lidar[label != 19]
label = label[label != 19]

# Quantize coordinates
coords = np.round(lidar[:, :3] / 0.05)
coords -= coords.min(0, keepdims=1)
feats = lidar

# Filter out duplicate points
coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
coords = torch.tensor(coords, dtype=torch.int)
feats = torch.tensor(feats[indices], dtype=torch.float)

inputs = SparseTensor(coords=coords, feats=feats)
inputs = sparse_collate([inputs]).cuda()


from model_zoo import spvnas_specialized

# Load the pre-trained model from model zoo
model = spvnas_specialized('SemanticKITTI_val_SPVNAS@65GMACs').cuda()
model.eval()

# Run the inference
outputs = model(inputs)
outputs = outputs.argmax(1).cpu().numpy()

# Map the prediction back to original points
outputs = outputs[inverse]


#%matplotlib inline

def configure_plotly_browser_state():
    import IPython
    from IPython.display import display
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))

import plotly
import plotly.graph_objs as go

trace = go.Scatter3d(
    x=lidar[:, 0],
    y=lidar[:, 1],
    z=lidar[:, 2],
    mode='markers',
    marker={
        'size': 1,
        'opacity': 0.8,
        'color': COLOR_MAP[outputs].tolist(),
    }
)

configure_plotly_browser_state()
plotly.offline.init_notebook_mode(connected=False)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2))
)

plotly.offline.iplot(go.Figure(data=[trace], layout=layout))

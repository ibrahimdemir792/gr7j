from pathlib import Path
import pandas as pd
import datetime
from input_data import InputDataHandler
from gr7j import ModelGr7j
import plotly.graph_objects as go
import spotpy
import json

data_path = Path('/home/ibrahim/gr7j/data')
df = pd.read_pickle(data_path / 'L0123001.pkl')
df.columns = ['date', 'precipitation', 'temperature', 'evapotranspiration', 'flow', 'flow_mm']
df.index = df['date']

inputs = InputDataHandler(ModelGr7j, df)
start_date = datetime.datetime(1989, 1, 1, 0, 0)
end_date = datetime.datetime(1999, 12, 31, 0, 0)
inputs = inputs.get_sub_period(start_date, end_date)

# Set the model :
# parameters = {
#         "X1": 170.532,
#         "X2": 0.457612,
#         "X3": 46.6777,
#         "X4": 3.73663,
#         "X5": 0.435072,
#         "X6": 17.8694,
#         "X7": 0.292645
#     }
with open ("parameters.json", "r") as file:
    parameters = json.load(file)

model = ModelGr7j(parameters)
model.set_parameters(parameters)  # Re-define the parameters for demonstration purpose.

# Initial state :
initial_states = {
    "production_store": 0.3, # was 0.5 according to Simon
    "routing_store": 0.5, # was 0.6 according to Simon
    "exponential_store": 0.3,
    "uh1": None,
    "uh2": None
}
model.set_states(initial_states)

outputs = model.run(inputs.data)
print(outputs.head())

nse = spotpy.objectivefunctions.nashsutcliffe(inputs.data['flow_mm'], outputs['flow'])
print(f"nse: {nse}")

fig = go.Figure([
    go.Scatter(x=outputs.index, y=outputs['flow'], name="Calculated"),
    go.Scatter(x=inputs.data.index, y=inputs.data['flow_mm'], name="Observed"),
])
fig.show()

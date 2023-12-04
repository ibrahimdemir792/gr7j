from pathlib import Path
import pandas as pd
import datetime
from input_data import InputDataHandler
from gr7j import ModelGr6j
import plotly.graph_objects as go

data_path = Path('/home/ibrahim/gr7j/data')
df = pd.read_pickle(data_path / 'L0123001.pkl')
df.columns = ['date', 'precipitation', 'temperature', 'evapotranspiration', 'flow', 'flow_mm']
df.index = df['date']

inputs = InputDataHandler(ModelGr6j, df)
start_date = datetime.datetime(1989, 1, 1, 0, 0)
end_date = datetime.datetime(1999, 12, 31, 0, 0)
inputs = inputs.get_sub_period(start_date, end_date)

# Set the model :
parameters = {
        "X1": 242.257,
        "X2": 0.637,
        "X3": 53.517,
        "X4": 2.218,
        "X5": 0.424,
        "X6": 4.759,
        "X7": 250
    }
model = ModelGr6j(parameters)
model.set_parameters(parameters)  # Re-define the parameters for demonstration purpose.

# Initial state :
initial_states = {
    "production_store": 0.5,
    "routing_store": 0.6,
    "exponential_store": 0.3,
    "uh1": None,
    "uh2": None
}
model.set_states(initial_states)

outputs = model.run(inputs.data)
print(outputs.head())

fig = go.Figure([
    go.Scatter(x=outputs.index, y=outputs['flow'], name="Calculated"),
    go.Scatter(x=inputs.data.index, y=inputs.data['flow_mm'], name="Observed"),
])
fig.show()

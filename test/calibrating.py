from pathlib import Path
import datetime
import pandas as pd
from numpy import sqrt, mean
import spotpy
import plotly.graph_objects as go
from input_data import InputDataHandler
from gr7j import ModelGr7j

data_path = Path('/home/ibrahim/gr7j/data')
df = pd.read_pickle(data_path / 'L0123001.pkl')
df.columns = ['date', 'precipitation', 'temperature', 'evapotranspiration', 'flow', 'flow_mm']
df.index = df['date']
print(df.head())

class SpotpySetup(object):
    """
    Interface to use the model with spotpy
    """

    def __init__(self, data):
        self.data = data
        self.model_inputs = InputDataHandler(ModelGr7j, self.data)
        self.params = [spotpy.parameter.Uniform('x1', 0.0, 2500.0),
                       spotpy.parameter.Uniform('x2', -5.0, 5.0),
                       spotpy.parameter.Uniform('x3', 0.0, 1000.0),
                       spotpy.parameter.Uniform('x4', 0.5, 10.0),
                       spotpy.parameter.Uniform('x5', -4.0, 4.0),
                       spotpy.parameter.Uniform('x6', 0.0, 20.0),
                       spotpy.parameter.Uniform('x7', 0.0, 1.0),
                       ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        simulations = self._run(x1=vector[0], x2=vector[1], x3=vector[2], x4=vector[3], x5=vector[4], x6=vector[5], x7=vector[6])
        return simulations

    def evaluation(self):
        return self.data['flow_mm'].values

    def objectivefunction(self, simulation, evaluation):
        res = - sqrt(mean((simulation - evaluation) ** 2.0))
        return res
    
    def _run(self, x1, x2, x3, x4, x5, x6, x7):
        parameters = {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "X6": x6, "X7": x7}
        model = ModelGr7j(parameters)
        outputs = model.run(self.model_inputs.data)
        return outputs['flow'].values
    

# Reduce the dataset to a sub period :
start_date = datetime.datetime(1997, 1, 1, 0, 0)
end_date = datetime.datetime(2008, 1, 1, 0, 0)
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
calibration_data = df.loc[mask]

spotpy_setup = SpotpySetup(calibration_data)

# sampler = spotpy.algorithms.mc(spotpy_setup, dbformat='ram')
sampler = spotpy.algorithms.mcmc(spotpy_setup, dbformat='ram')
# sampler = spotpy.algorithms.mle(spotpy_setup, dbformat='ram')
# sampler = spotpy.algorithms.lhs(spotpy_setup, dbformat='ram')
# sampler = spotpy.algorithms.sceua(spotpy_setup, dbformat='ram')
# sampler = spotpy.algorithms.demcz(spotpy_setup, dbformat='ram')
# sampler = spotpy.algorithms.sa(spotpy_setup, dbformat='ram')
# sampler = spotpy.algorithms.rope(spotpy_setup, dbformat='ram')

sampler.sample(2000)
results=sampler.getdata() 
best_parameters = spotpy.analyser.get_best_parameterset(results)
print(best_parameters)


parameters = list(best_parameters[0])
parameters = {"X1": parameters[0], "X2": parameters[1], "X3": parameters[2], "X4": parameters[3], "X5": parameters[4], "X6": parameters[5], "X7": parameters[6]}

start_date = datetime.datetime(1989, 1, 1, 0, 0)
end_date = datetime.datetime(1999, 12, 31, 0, 0)
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
validation_data = df.loc[mask]

model = ModelGr7j(parameters)
outputs = model.run(validation_data)

# Remove the first year used to warm up the model :
filtered_input = validation_data[validation_data.index >= datetime.datetime(1990, 1, 1, 0, 0)]
filtered_output = outputs[outputs.index >= datetime.datetime(1990, 1, 1, 0, 0)]

rmse = sqrt(mean((filtered_output['flow'] - filtered_input['flow_mm'].values) ** 2.0))
print(f"rmse: {rmse}")

nse = spotpy.objectivefunctions.nashsutcliffe(filtered_output['flow'], filtered_input['flow_mm'])
print(f"nse: {nse}")

# fig = go.Figure([
#     go.Scatter(x=filtered_output.index, y=filtered_output['flow'], name="Calculated"),
#     go.Scatter(x=filtered_input.index, y=filtered_input['flow_mm'], name="Observed"),
# ])
# fig.show()


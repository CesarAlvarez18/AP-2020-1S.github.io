import dash
import dash_core_components as dcc 
import dash_html_components as html

from sodapy import Socrata
import numpy as np 
import re
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd 
import random
import math
import time
import datetime as dt
from datetime import timedelta
import operator 
plt.style.use('fivethirtyeight')
#%matplotlib inline
#import warnings
#warnings.filterwarnings("ignore")
#!pip install plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
#!pip install pyramid-arima
#from pyramid.arima import auto_arima
std=StandardScaler()
#pd.set_option('display.float_format', lambda x: '%.6f' % x)


covid_19  = Socrata("www.datos.gov.co", None)
results = covid_19.get_all("gt2j-8ykr", limit=800000)


results_df = pd.DataFrame.from_records(results)
#results_df.describe()

df = (results_df[results_df["estado"] != "N/A"])[["id_de_caso", "fecha_de_notificaci_n", "ciudad_de_ubicaci_n", "atenci_n", "edad", "sexo", "tipo", "estado", "fis", "fecha_diagnostico", "fecha_recuperado", "fecha_reporte_web", "tipo_recuperaci_n", "fecha_de_muerte"]]
df.columns = ["ID_de_caso", "Fecha_de_notificación", "Ciudad_de_ubicación", "atención", "Edad", "Sexo", "Tipo", "Estado", "FIS", "Fecha_diagnostico", "Fecha_recuperado", "fecha_reporte_web", "Tipo_recuperación", "Fecha_de_muerte"]
df = df[df.Ciudad_de_ubicación.isin(["Bogotá D.C.", "Medellín", "Barranquilla", "Cali", "Cartagena de Indias"])]
new_df = df[["fecha_reporte_web", "atención", "Ciudad_de_ubicación"]]
dateRegex = re.compile(r"[a-zA-Z]\d\d\:\d\d\:\d\d\.\d\d\d")
new_df = new_df.replace(dateRegex, "",regex=True, inplace=False)
new_df['fecha_reporte_web'] = pd.to_datetime(new_df['fecha_reporte_web'])
new_df["atención"] = new_df["atención"].replace(to_replace=["Casa", "Hospital", "Hospital UCI", "N/A"], value="ACTIVOS")
new_df["atención"] = new_df["atención"].replace(to_replace=["Recuperado"], value="RECUPERADOS")
new_df["atención"] = new_df["atención"].replace(to_replace=["Fallecido"], value="MUERTES")
confirmados = new_df
recuperados = new_df[new_df["atención"] == "RECUPERADOS"]
activos = new_df[new_df["atención"] == "ACTIVOS"]
muertos = new_df[new_df["atención"] == "MUERTES"]
new_df["Confirmados"]=1
new_df["Recuperados"]= np.where(new_df["atención"] == "RECUPERADOS", 1, 0)
new_df["Muertos"]= np.where(new_df["atención"] == "MUERTES", 1, 0)
new_df["Activos"]= np.where(new_df["atención"] == "ACTIVOS", 1, 0)

new_df["fecha_reporte_web"]=pd.to_datetime(new_df["fecha_reporte_web"])
df_agrupado= new_df.groupby(["Ciudad_de_ubicación","fecha_reporte_web"]).agg({"Confirmados":'sum',"Recuperados":'sum',"Activos":'sum',"Muertos":'sum'})

df_agrupado2= new_df.groupby(["fecha_reporte_web"]).agg({"Confirmados":'sum',"Recuperados":'sum',"Activos":'sum',"Muertos":'sum'})

df_agrupado2["Confirmados_acum"] =df_agrupado2["Confirmados"].cumsum()
df_agrupado2["Recuperados_acum"] =df_agrupado2["Recuperados"].cumsum()
df_agrupado2["Activos_acum"] =df_agrupado2["Activos"].cumsum()
df_agrupado2["Muertos_acum"] =df_agrupado2["Muertos"].cumsum()

df_agrupado2["Días"]=df_agrupado2.index-df_agrupado2.index.min()
df_agrupado2["Días"]=df_agrupado2["Días"].dt.days



df_agrupado2["Semana"]=df_agrupado2.index.weekofyear

semana_num=[]
semana_confirmados=[]
semana_recuperados=[]
semana_muertos=[]
w=1
for i in list(df_agrupado2["Semana"].unique()):
    semana_confirmados.append(df_agrupado2[df_agrupado2["Semana"]==i]["Confirmados"].iloc[-1])
    semana_recuperados.append(df_agrupado2[df_agrupado2["Semana"]==i]["Recuperados"].iloc[-1])
    semana_muertos.append(df_agrupado2[df_agrupado2["Semana"]==i]["Muertos"].iloc[-1])
    semana_num.append(w)
    w=w+1


app = dash.Dash()

app.layout = html.Div(children=[
    html.H1('Dash Covid 19 en Colombia'),
    dcc.Graph(id='example',
    figure ={
        'data': [
            {'x' : semana_num, 'y':semana_confirmados, 'type':'line', 'name':'Crecimiento semanal casos confirmados'},
            {'x' : semana_num, 'y':semana_recuperados, 'type':'line', 'name': 'Crecimiento semanal casos recuperados'},
            {'x' : semana_num, 'y':semana_muertos, 'type':'line', 'name': 'Crecimiento semanal muertos'}
        ],
        'layout': {
            'title': 'Basic Dash Example'
        }
    })
])



if __name__ == '__main__':
    app.run_server(debug=True)


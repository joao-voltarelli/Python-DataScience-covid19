import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dados_covid = pd.read_csv('COVID19_state.csv')

#REMOVENDO DADOS QUE NÃO SERÃO RELEVANTES PARA O NOSSO MODELO
dados_covid.drop(['Sex Ratio', 'Smoking Rate', 'Flu Deaths', 'Respiratory Deaths', 'Physicians', 'Pollution', 'Med-Large Airports', 'Temperature', 'Urban', 'Age 0-25', 'Age 26-54', 'Age 55+', 'School Closure Date'], axis=1, inplace=True)

#CORRELACAO DOS DADOS UTILIZANDO MAPA DE CALOR
plt.figure(figsize=(15, 15))
sns.heatmap(dados_covid.corr(), annot=True, cmap="Blues")

#MATRIZ DE CORRELACAO
dados_covid.corr()
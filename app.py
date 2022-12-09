""" Construindo um frontend com Streamlit """
from sklearn import metrics, model_selection, tree
import streamlit as st
import pandas as pd


# Título
st.markdown(
    "<h1 style='text-align: center; color: grey;'>Análise de Câncer</h1>",
    unsafe_allow_html=True
)

# Dataset
df = pd.read_csv('dataR2.csv')

# Cabeçalho
st.subheader('Informações dos dados')

# Nome do paciente
user_input = st.sidebar.text_input('Digite seu nome ')

# Dados de entrada
X = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
y = df['Classification']

# Dados de entrada
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Dados dos usuários com a função
def get_user_data():
    """ Função para receber os dados do usuário """
    idade           = st.sidebar.slider('Idade', 24, 89, 24)
    bmi             = st.sidebar.slider('BMI', 18, 39, 18)
    glicose         = st.sidebar.slider('Glicose', 60, 201, 60)
    insulina        = st.sidebar.slider('Insulina', 2, 59, 2)
    homa            = st.sidebar.slider('HOMA', 0, 25, 0)
    leptina         = st.sidebar.slider('Leptina', 4, 90, 4)
    adiponectin     = st.sidebar.slider('Adiponectina', 1, 38, 1)
    resistina       = st.sidebar.slider('Resistina', 3, 82, 3)
    mcp1            = st.sidebar.slider('MCP.1', 45, 1699, 45)

    paciente_dados = {
        'Age': idade,
        'BMI': bmi,
        'Glucose': glicose,
        'Insulin': insulina,
        'HOMA': homa,
        'Leptin': leptina,
        'Adiponectin': adiponectin,
        'Resistin': resistina,
        'MCP.1': mcp1
    }

    features = pd.DataFrame(paciente_dados, index=[0])

    return features

paciente_entrada_variaveis = get_user_data()

grafico = st.bar_chart(paciente_entrada_variaveis)

st.subheader('Dados do paciente')
st.write(paciente_entrada_variaveis)

dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=101)
dtc.fit(X_train, y_train)

# Acurácia do modelo
st.subheader('Acurácia do modelo')
st.write(round(metrics.accuracy_score(y_test, dtc.predict(X_test)) * 100, 2))

# Previsão
predict = dtc.predict(paciente_entrada_variaveis)

st.subheader('Previsão: ')

def predizer(modelo_predizado):
    """ Função para predizer """

    if modelo_predizado[0] == 1:
        return 'Paciente saudável'
    return 'Paciente diagnosticado com câncer'


st.write(predizer(predict))

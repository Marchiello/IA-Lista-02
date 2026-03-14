import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Testei pra visualizar num gráfico 3D mas não adiantou muita coisa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------- 1.Carregamento dos Dados:

# Importar o dataset
df = pd.read_csv('dataset_emprestimo_aprovacao.csv')

# Exibir as primeiras linhas
df.head()
df.tail()

df.sort_values(["renda_mensal"], ascending=[False])

# ----------------------------- 2.Analise Exploratoria:

# Análise exploratória

def gerar_grafico(campo_1, campo_2, campo_base):
  plt.scatter(df[campo_1], df[campo_2], c=df[campo_base].map({0: "red", 1: 'green'}))
  plt.xlabel(campo_1)
  plt.ylabel(campo_2)
  plt.title("Metricas de aprovacao de emprestimo")
  plt.show()

# Buscando encontrar relação entre renda mensal e score de crédito para a aprovação do empréstimo
gerar_grafico("renda_mensal", "score_credito", "emprestimo_aprovado")

print("\n\n")

# Buscando encontrar relação entre renda mensal e numero de dividas ativas para a aprovação do empréstimo
gerar_grafico("renda_mensal", "dividas_ativas", "emprestimo_aprovado")

print("\n\n")

# Buscando encontrar relação entre score de credito e numero de dividas ativas para a aprovação do empréstimo
gerar_grafico("score_credito", "dividas_ativas", "emprestimo_aprovado")


print('\n\n')

# Verificar possiveis relações entre as variáveis e a aprovação do empréstimo
df.corr()


# ----------------------------- 3.Preparacao dos Dados:

# Separar as variáveis independentes(X) da variavel alvo(Y)

x = df[['renda_mensal', 'score_credito', 'dividas_ativas']]
y = df[['emprestimo_aprovado']]


# Dividir o dataset em treino e teste

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=45
)

# print(x_test, y_test)

# ----------------------------- 4.Treinamento do Modelo:

# Treinamento do Modelo
modelo = LogisticRegression()
modelo.fit(x_train, y_train)

# Previsão do modelo

y_pred = modelo.predict(x_test)

print(f"Previsões")
print(f"{y_pred}")
print(f"{y_test}")

# Matriz de confusão

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Reprovado", "Aprovado"],
    cmap="Blues"
)

plt.title("Matriz de confusão")
plt.show()

# ----------------------------- 5.Metricas de avaliacao do modelo:

# Acurácia

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc}")

# Relatório de classificação
print(f"Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Teste individual de previsão
nova_tentativa = pd.DataFrame(
    {
        "renda_mensal": [0],
        "score_credito": [500],
        "dividas_ativas": [4]
    }
)

previsão = modelo.predict(nova_tentativa)
print(f"Previsão: {previsão}")
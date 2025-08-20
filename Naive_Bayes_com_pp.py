# ====================================================================================================
# Nome do Arquivo : Naive_Bayes.py
# Autores         : Débora Leandro de Andrade e Juan Diego de Paula Rollemberg
# Curso           : PES - Colaborador Embraer
# Disciplina      : Aprendizagem de máquina I
# Professor       : George Darmilton
# Data            : 03/08/2025                                                
# ====================================================================================================

"""
Esse script tem como objetivo ler o dataset selecionado e aplicar o algoritmo Naive Bayes.
É realizado um treinamento considerando todos os atributos.       
"""

#%% Imports ==========================================================================================

# Utilidades
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Classes do modelo de aprendizado
from sklearn.naive_bayes import GaussianNB

# Função para importar o dataset
from sklearn.datasets import fetch_openml

# Funções de avaliação dos modelos
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (classification_report, 
                            confusion_matrix, 
                            ConfusionMatrixDisplay, 
                            roc_curve, auc)

from sklearn.model_selection import train_test_split, KFold, cross_val_score



#%% Inicio do programa ==========================================================================================
# Lê o dataset como dataframe
dataset_sklearn = fetch_openml(name="air-quality-and-pollution-assessment", version=1, as_frame=True)
dataset = dataset_sklearn.frame

#%% Pré-visualização do dataset
dataset.head()

#%% Resumo do dataset
dataset.describe()

#%% Mapeando os valores da classe para inteiro (para fins de visualização da região de decisão)
dataset['Air_Quality'] = pd.factorize(dataset['Air_Quality'])[0]

#%% Matriz de correlação - Todos os atributo
correlation_matrix = dataset.corr()

plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 14},
)

plt.title("Correlação com a variável alvo", fontsize=20)
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Aumentar fonte da legenda (colorbar)
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=14)

plt.show()

#%% Primeiro treinamento ==========================================================================================
# Todos os atributos são considerados
# -----------------------------------------------------------

# Define os atributos e a variável alvo
target_column='Air_Quality'
X = dataset.drop(columns=target_column)
y = dataset[target_column]

# Define a semente
seed=42

# Divide o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

#%% Faz o treinamento do modelo de Naive Bayes
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)

y_pred = NB_model.predict(X_test)


#%% Avaliacao: #1 Matriz de confusao
cm = confusion_matrix(y_test, y_pred, labels=NB_model.classes_)
print(cm)

print(f"\n Labels:{NB_model.classes_} \n")

#display_labels - define como será a ordem das classes na matriz
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=NB_model.classes_)
disp_cm.plot()

#%% Avaliacao: #2 Curvas ROC

# Obtem as probabilidades preditas
y_pred_prob = NB_model.predict_proba(X_test)

# Ordena as classes para a binarização
classes = sorted(y.unique())

# Transforma o vetor teste em formato binário
y_test_bin = label_binarize(y_test, classes=classes)

# Calcula as curvas ROC para cada classe
plt.figure(figsize=(10, 8))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Classe {classes[i]} (AUC = {roc_auc:.2f})', linewidth=3)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('Taxa de Falsos Positivos', fontsize=14)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
plt.title('Curvas ROC - Naive Bayes por Classe', fontsize=16)
plt.legend(loc='lower right',fontsize=12)
plt.grid(True)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


#%% Validação: #1 Validação Cruzada - K Fold, K=10

kf = KFold(n_splits=10)

scores = cross_val_score(NB_model, X, y, cv=kf, scoring='accuracy')

print("Acurácias por fold:", scores)
print("Acurácia média:", scores.mean())
print("K-fold: %.3f ± %.3f" % (scores.mean(), scores.std()))

#%% Avaliação: #4 Relatório de classificação e métricas
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-1]  # Remover 'accuracy' e 'macro avg'

plt.figure(figsize=(10, 6))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Precisão, Recall e F1-Score por Classe - Naive Bayes")
plt.xlabel("Classe")
plt.ylabel("Valor")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()


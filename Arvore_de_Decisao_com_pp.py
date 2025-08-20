# ====================================================================================================
# Nome do Arquivo : Arvore_de_Decisao.ipynb
# Autores         : Débora Leandro de Andrade e Juan Diego de Paula Rollemberg
# Curso           : PES - Colaborador Embraer
# Disciplina      : Aprendizagem de máquina I
# Professor       : George Darmilton
# Data            : 03/08/2025
# ====================================================================================================

#Esse script tem como objetivo ler o dataset selecionado e aplicar o algoritmo Árvore de Decisão.
#É realizado um treinamento considerando todos os atributos.

#instalação das bibliotecas de manipualção e visualização de dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from sklearn.pipeline import Pipeline

#acesso ao dataset
from sklearn.datasets import fetch_openml

#classes do modelo de aprendizado
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

#funções de avaliação dos modelos
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

# Funções de pre processamento
from sklearn.feature_selection import VarianceThreshold  # Selecao de atributos

# Função para pre-processamento: selecao de prototipos
from imblearn.under_sampling import EditedNearestNeighbours

# Função para pre-processamento: balanceamento 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour

# Função para pre-processamento: redução de dimensionalidade
from sklearn.decomposition import PCA

# Função para pre-processamento: Escalonamento
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

#%% Treinamento Original - Aprendizagem de Maquina I =============

#fetch dataset
X,y = fetch_openml(data_id=46880, return_X_y=True)

dataset = pd.concat([X,y], axis=1)

dataset.head()

#%%
#definindo semente
seed=42

# Seleciona todos os campos menos a classe alvo para a variável "X".
X = dataset.iloc[:,:-1]
y = dataset.loc[:,"Air_Quality"]

#separando o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
print(X_train.shape)
print(X_test.shape)

#%%
#loop para teste utilizando diversos criterion e max_depth
for i, criterion in enumerate(['gini', 'entropy']):

    for j, max_depth in enumerate([3,4,5]):

      print(criterion.upper())
      print(f"MAX_DEPTH - {max_depth} \n ")

      model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=seed)

      #treinando o modelo
      model.fit(X_train, y_train)

      #predição
      y_pred = model.predict(X_test)

      #relatório do classificador
      print(classification_report(y_test, y_pred))

      #calcula a matriz de confusão de acordo com os parâmetros acima
      cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
      disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)

      disp_cm.plot()

      plt.show()

      print('\n')

      #10-fold cross validation
      kf = KFold(n_splits=10)

      scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

      print("Acurácias por fold:", scores)
      print("Acurácia média:", scores.mean())
      print("K-fold: %.3f ± %.3f" % (scores.mean(), scores.std()))
      print('\n')

      #calculo das curvas ROC para cada classe
      y_pred_prob = model.predict_proba(X_test)

      classes = sorted(y.unique())

      y_test_bin = label_binarize(y_test, classes=classes)

      plt.figure(figsize=(10, 8))
      for i in range(len(classes)):
          fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
          roc_auc = auc(fpr, tpr)
          plt.plot(fpr, tpr, label=f'Classe {classes[i]} (AUC = {roc_auc:.2f})', linewidth=3)

      plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
      plt.xlabel('Taxa de Falsos Positivos', fontsize=14)
      plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
      plt.title('Curvas ROC - Árvore de Decisão por Classe', fontsize=16)
      plt.legend(loc='lower right',fontsize=12)
      plt.grid(True)

      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)

      plt.show()


#%% Treinamento com pré-processamento - Aprendizagem de Máquina II ==========================================================================================
# Selecao de atributos

filter_variance = VarianceThreshold(0.01)
X_filtered = filter_variance.fit_transform(X)

# Atributos removidos
print("Número inicial de features: %d" %(X.shape[1]))
print("Features selecionadas: %d" %(X_filtered.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, stratify=y, random_state=42)
model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# %%

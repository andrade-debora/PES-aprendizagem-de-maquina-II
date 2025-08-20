# ====================================================================================================
# Nome do Arquivo : MLP.ipynb
# Autores         : Débora Leandro de Andrade e Juan Diego de Paula Rollemberg
# Curso           : PES - Colaborador Embraer
# Disciplina      : Aprendizagem de máquina I
# Professor       : George Darmilton
# Data            : 03/08/2025
# ====================================================================================================

#Esse script tem como objetivo ler o dataset selecionado e aplicar o algoritmo MLP.
#É realizado um treinamento considerando todos os atributos.

#%%
#instalação das bibliotecas de manipualção e visualização de dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import seaborn as sns

#acesso ao dataset
from sklearn.datasets import fetch_openml

#classes do modelo de aprendizado
from sklearn.neural_network import MLPClassifier

#funções de avaliação dos modelos
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

import warnings
warnings.filterwarnings('ignore')

#%%
#fetch dataset
X,y = fetch_openml(data_id=46880, return_X_y=True)

dataset = pd.concat([X,y], axis=1)

dataset.head()

#%%
#definindo semente
seed=42

#seleciona todos os campos menos a classe alvo para a variável "X".
X = dataset.iloc[:,:-1]
y = dataset.loc[:,"Air_Quality"]

#separando o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
print(X_train.shape)
print(X_test.shape)

#%%
mlp = MLPClassifier(random_state=seed)

param_distributions = {
      'hidden_layer_sizes': [(64,), (128,), (64, 128), (32, 64, 128), (64, 128, 32)],
      'activation': ['logistic', 'tanh', 'relu'],
      'solver': ['lbfgs', 'sgd', 'adam'],
      'alpha': uniform(loc=0.0001, scale=0.01),
      'learning_rate': ['constant', 'adaptive'],
      'max_iter': randint(low=100, high=500)
}

random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions,
        n_iter=20,
        cv=10,
        scoring='accuracy',
        random_state=seed,
        n_jobs=-1,
        return_train_score=True
)

random_search.fit(X_train, y_train)

#acessa o dicionário cv_results_
results_dict = random_search.cv_results_

#converte para DataFrame para uma melhor leitura
results_df = pd.DataFrame(results_dict)

print("Best parameters found:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

results_df

#%%
#definindo o modelo com os melhores parâmetros
model = MLPClassifier(random_state=seed, **random_search.best_params_)

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

plt.plot(model.loss_curve_)

print('\n')

#10-fold cross validation
kf = KFold(n_splits=10)

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("Acurácias por fold:", scores)
print("Acurácia média:", scores.mean())
print("K-fold: %.3f ± %.3f" % (scores.mean(), scores.std()))
print('\n')

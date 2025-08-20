# ====================================================================================================
# Nome do Arquivo : KNN.py
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
from sklearn.pipeline import Pipeline

# Classes do modelo de aprendizado
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

# Acesso ao dataset
from sklearn.datasets import fetch_openml

# Funções de avaliação dos modelos
from sklearn.metrics import (classification_report, 
                             ConfusionMatrixDisplay, 
                             confusion_matrix, 
                             roc_auc_score, 
                             roc_curve, auc, 
                             accuracy_score,
                             precision_score)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

# Funções para pre-processamento: selecao de atributos
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# Função para pre-processamento: selecao de prototipos
from imblearn.under_sampling import EditedNearestNeighbours

# Função para pre-processamento: balanceamento 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour

# Função para pre-processamento: redução de dimensionalidade
from sklearn.decomposition import PCA

# Função para pre-processamento: Escalonamento
from sklearn.preprocessing import StandardScaler

# Função de busca por melhores parametros
from sklearn.model_selection import GridSearchCV


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
''' 
Moderate - 0
Good  - 1 
Hazardous - 2 
Poor - 3
'''


   
#%% Treinamento Original - Aprendizagem de Máquina I ==========================================================================================
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

print("Shape conjunto de treino:", X_train.shape)
print("Shape conjunto de teste:", X_test.shape)

model = KNeighborsClassifier(n_neighbors=10, metric='manhattan')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

#%% Treinamento com pré-processamento - Aprendizagem de Máquina II ==========================================================================================

#%% Seleção de atributos
base_estimator = SVC(kernel="linear")
rfe = RFE(base_estimator, n_features_to_select=2, step=1)
X_filtered = rfe.fit_transform(X, y)

print(X_filtered.shape)

#Nomes das features vistas durante o ajuste
ft_names = rfe.feature_names_in_
print(ft_names)

#Máscara de features selecionadas.
print(rfe.support_) 
print("\nFeatures selecionadas:")
print(ft_names[rfe.support_])

# Dataset com atributos selecionados
X = dataset[['CO','Proximity_to_Industrial_Areas']]
y = dataset['Air_Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
X_train.shape

model = KNeighborsClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

#%% Seleção de protótipos
#
#def percentage(train, resampled):
#  excluidos = (train-resampled)
#  percentage = 100 * float(excluidos)/float(train)
#  return percentage
#
#enn = EditedNearestNeighbours()
#enn.get_params()
#
#X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)
#
#y_train_enn.value_counts()
#
#print("Porcentagem de redução: %.2f%%" % (percentage(y_train.count(), y_train_enn.count())))
#
#model = KNeighborsClassifier()
#
#model.fit(X_train_enn, y_train_enn)
#
#y_pred = model.predict(X_test)
#
#print(classification_report(y_test, y_pred))
#
##%%=================================================

#%% Balanceamento - Undersampling vs. Oversampling

pipeline = Pipeline([('knn', KNeighborsClassifier())])

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
score_rus = cross_val_score(pipeline, X_rus, y_rus, cv=5, scoring='accuracy').mean()

# CNN Undersampling
cnn = CondensedNearestNeighbour()
X_cnn, y_cnn = cnn.fit_resample(X, y)
score_cnn = cross_val_score(pipeline, X_cnn, y_cnn, cv=5, scoring='accuracy').mean()

# SMOTE Oversampling
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
score_smote = cross_val_score(pipeline, X_smote, y_smote, cv=5, scoring='accuracy').mean()

# Exibir resultados
print(f"Acurácia média com Random Undersampling: {score_rus:.4f}")
print(f"Acurácia média com CNN Undersampling: {score_cnn:.4f}")
print(f"Acurácia média com SMOTE: {score_smote:.4f}")

# Aplicando o melhor método (SMOTE)
smt = SMOTE(random_state=seed) # k_neighbors por padrão é 5

#Aplicando RandomUnderSampler
X_smt, y_smt = smt.fit_resample(X, y)

# Novo dataset balanceado
df_SMOTE = pd.DataFrame(X_smt, columns=X_train.columns)
df_SMOTE.insert(0, 'Air_Quality', y_smt)  

print(df_SMOTE.shape)

sns.barplot(x="Air_Quality", y="Air_Quality",  data=df_SMOTE,  estimator=lambda x: len(x) / len(df_SMOTE) * 100)

#Separando o conjunto de dados em treinamento e teste
X_train_smt, X_test_smt, y_train_smt, y_test_smt = train_test_split(X_smt, y_smt, test_size=0.3, stratify=y_smt, random_state=seed)

print("Shape de X_train:", X_train_smt.shape)
print("Shape de X_test:", X_test_smt.shape)
print("Shape de y_train:", y_train_smt.shape)
print("Shape de y_test:", y_test_smt.shape)

model = KNeighborsClassifier()

#treinando o modelo
model.fit(X_train_smt, y_train_smt)

#predição
y_pred_smt = model.predict(X_test_smt)

#Resultados do classificador
print(classification_report(y_test_smt, y_pred_smt))

#%% Utilizando pipeline, PCA e gridsearch
# Escalonando antes de aplicar o PCA
scaling = StandardScaler()

scaling.fit(X_train_smt)

X_train_ss = scaling.transform(X_train_smt)
X_test_ss = scaling.transform(X_test_smt)

# Pipeline com scaler, PCA e KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

# Parâmetros para o GridSearch
param_grid = {
    'pca__n_components': [1, 2 , 3],  # ajuste conforme o número de features
    'knn__n_neighbors': [3,5,7,9,15,20,25],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev']
}
#%%
# GridSearch com validação cruzada
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_smt, y_smt)

# Resultados
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor acurácia:", grid_search.best_score_)

pca = PCA(n_components=grid_search.best_params_['pca__n_components'])
pca.fit(X_train_smt)

X_train_pca = pca.transform(X_train_smt)
X_test_pca = pca.transform(X_test_smt)

# definir um modelo
model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['knn__n_neighbors'],
                             weights=grid_search.best_params_['knn__weights'],
                             metric=grid_search.best_params_['knn__metric'])

model.fit(X_train_pca, y_train_smt)

y_pred = model.predict(X_test_pca)

print(classification_report(y_test_smt, y_pred))


#%% Validação: #1 Validação Cruzada - K Fold, K=10

kf = KFold(n_splits=10)

scores = cross_val_score(model, X_smt, y_smt, cv=kf, scoring='accuracy')

print("Acurácias por fold:", scores)
print("Acurácia média:", scores.mean())
print("K-fold: %.3f ± %.3f" % (scores.mean(), scores.std()))

#%% Avaliação: #4 Relatório de classificação e métricas
report = classification_report(y_test_smt, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-1]  # Remover 'accuracy' e 'macro avg'

plt.figure(figsize=(10, 6))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Precisão, Recall e F1-Score por Classe - KNN, N = 10")
plt.xlabel("Classe")
plt.ylabel("Valor")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()


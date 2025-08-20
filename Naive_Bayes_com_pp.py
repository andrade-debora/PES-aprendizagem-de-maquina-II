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
from sklearn.pipeline import Pipeline

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

# Função de busca por melhores parametros
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

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

#%% Pré-processamento | Seleção de atributos ===========================

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



#%% Pré-processamento | Seleção de protótipos ===========================

def percentage(train, resampled):
  excluidos = (train-resampled)
  percentage = 100 * float(excluidos)/float(train)
  return percentage

enn = EditedNearestNeighbours()
enn.get_params()

X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

y_train_enn.value_counts()

print("Porcentagem de redução: %.2f%%" % (percentage(y_train.count(), y_train_enn.count())))

model = GaussianNB()

model.fit(X_train_enn, y_train_enn)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

#%% Pré-processamento | Balanceamento (Under ou oversampling) ===========================

# Busca o melhor método
pipeline = Pipeline([('NB', GaussianNB())])

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

model = GaussianNB()

#treinando o modelo
model.fit(X_train_smt, y_train_smt)

#predição
y_pred_smt = model.predict(X_test_smt)

#Resultados do classificador
print(classification_report(y_test_smt, y_pred_smt))

#%% Pré-processamento | Gridsearch e diminuição da dimensionalidade (PCA) ===========================

scaling = StandardScaler()

scaling.fit(X_train_smt)

X_train_ss = scaling.transform(X_train_smt)
X_test_ss = scaling.transform(X_test_smt)

# Pipeline com scaler, PCA e NB
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('nb', GaussianNB())
])

# Parâmetros para o GridSearch
param_grid = {
    'pca__n_components': [2, 5, 10, 15]
}


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
model = GaussianNB()

model.fit(X_train_pca, y_train_smt)

y_pred = model.predict(X_test_pca)

print(classification_report(y_test_smt, y_pred))
#%% Avaliacao: #1 Matriz de confusao
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)

print(f"\n Labels:{model.classes_} \n")

#display_labels - define como será a ordem das classes na matriz
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp_cm.plot()

#%% Avaliacao: #2 Curvas ROC

# Obtem as probabilidades preditas
y_pred_prob = model.predict_proba(X_test)

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

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

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


# ====================================================================================================
# Nome do Arquivo : Regressao_Logistica.ipynb
# Autores         : Débora Leandro de Andrade e Juan Diego de Paula Rollemberg
# Curso           : PES - Colaborador Embraer
# Disciplina      : Aprendizagem de máquina I
# Professor       : George Darmilton
# Data            : 03/08/2025
# ====================================================================================================

#Esse script tem como objetivo ler o dataset selecionado e aplicar o algoritmo Regressão Logística.
#É realizado um treinamento considerando todos os atributos.

#%% instalação das bibliotecas de manipualção e visualização de dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from sklearn.pipeline import Pipeline

#acesso ao dataset
from sklearn.datasets import fetch_openml

#classes do modelo de aprendizado
from sklearn.linear_model import LogisticRegression

#funções de avaliação dos modelos
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

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

import warnings
warnings.filterwarnings('ignore')

#%%
#fetch dataset
X,y = fetch_openml(data_id=46880, return_X_y=True)

dataset = pd.concat([X,y], axis=1)

dataset.head()

#definindo semente
seed=42

#seleciona todos os campos menos a classe alvo para a variável "X".
X = dataset.iloc[:,:-1]
y = dataset.loc[:,"Air_Quality"]

#%% Pré-processamento | Seleção de atributos ===========================
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

model = LogisticRegression(random_state=seed)

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

model = LogisticRegression(random_state=seed)

model.fit(X_train_enn, y_train_enn)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


#%% Pré-processamento | Balanceamento (Under ou oversampling) ===========================

# Busca o melhor método
pipeline = Pipeline([('RL', LogisticRegression(random_state=seed)
)])

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

model.fit(X_train_enn, y_train_enn)

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

# Pipeline com scaler, PCA e LR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression(max_iter=1000))
])



# Parâmetros para o GridSearch
param_grid = {
    'pca__n_components': [2, 3, 5],
    'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'clf__solver': ['saga'],
    'clf__C': [0.1, 1.0, 10.0]
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
model = LogisticRegression(penalty = grid_search.best_params_['clf__penalty'],
                           solver = grid_search.best_params_['clf__solver'],
                           C = grid_search.best_params_['clf__C'],
                           random_state = seed)
model.fit(X_train_pca, y_train_smt)

y_pred = model.predict(X_test_pca)

print(classification_report(y_test_smt, y_pred))

#%% Avaliacao: #1 Matriz de confusao
cm = confusion_matrix(y_test_smt, y_pred, labels=model.classes_)
print(cm)

print(f"\n Labels:{model.classes_} \n")

#display_labels - define como será a ordem das classes na matriz
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp_cm.plot()

#%% Avaliacao: #2 Curvas ROC

# Obtem as probabilidades preditas
y_pred_prob = model.predict_proba(X_test_smt)

# Ordena as classes para a binarização
classes = sorted(y.unique())

# Transforma o vetor teste em formato binário
y_test_bin = label_binarize(y_test_smt, classes=classes)

# Calcula as curvas ROC para cada classe
plt.figure(figsize=(10, 8))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Classe {classes[i]} (AUC = {roc_auc:.2f})', linewidth=3)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('Taxa de Falsos Positivos', fontsize=14)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
plt.title('Curvas ROC - Regressao Logistica por Classe', fontsize=16)
plt.legend(loc='lower right',fontsize=12)
plt.grid(True)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

#%% Validação: #1 Validação Cruzada - K Fold, K=10

kf = KFold(n_splits=10)

scores = cross_val_score(model, X_smote, y_smote, cv=kf, scoring='accuracy')

print("Acurácias por fold:", scores)
print("Acurácia média:", scores.mean())
print("K-fold: %.3f ± %.3f" % (scores.mean(), scores.std()))

#%% Avaliação: #4 Relatório de classificação e métricas
report = classification_report(y_test_smt, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-1]  # Remover 'accuracy' e 'macro avg'

plt.figure(figsize=(10, 6))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Precisão, Recall e F1-Score por Classe - Regressao Logistica")
plt.xlabel("Classe")
plt.ylabel("Valor")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

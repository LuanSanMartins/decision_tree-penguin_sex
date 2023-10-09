# -*- coding: utf-8 -*-
"""
Created on Wed May  3 00:02:54 2023

@author: Luan
"""

# =============================================================================
# ========================== Importar as Bibliotecas ==========================
# =============================================================================

from os import getcwd, sep
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# =============================================================================

warnings.filterwarnings('ignore')

# =============================================================================
# ============================== Primeira Etapa ===============================
# =============================================================================

# ============================== Dataset Inicial ==============================

# Ler o dataset inicial
df0 = pd.read_csv(getcwd()+sep+"penguins.csv")

# Verificar se há valores faltantes no dataset inicial
print("\nValores Faltantes no Dataset Inicial: \n\n", df0.isnull().sum())

# Verificar a estatística descritiva no dataset inicial
print("\nEstatística Descrtivia do Dataset Inicial: \n\n", df0.describe())

# Calcular correlação dos dados no dataset inicial
correlacao0 = df0.corr(method ='pearson')

# Exibir o cálculo da correlação dos dados no dataset inicial
print("\nCorrelação do Dataset Inicial: \n\n", correlacao0)

# ======================= Tratamento do dataset inicial =======================

# Excluir Valores faltantes no dataset inicial
df = df0.dropna()

# ============================== Dataset Tratado ==============================

# Verificar se há valores faltantes no dataset tratado
print("\nValores Faltantes no Dataset Tratado: \n\n", df.isnull().sum())

# Verificar a estatística descritiva no dataset tratado
print("\nEstatística Descrtivia do Dataset Tratado: \n\n", df.describe())

# Calcular correlação dos dados no dataset tratado
correlacao = df.corr(method ='pearson')

# Exibir o cálculo da correlação dos dados no dataset tratado
print("\nCorrelação do Dataset Inicial: \n\n", correlacao)

# ============================= Pré-Processamento =============================

# Substituir as variáveis categóricas nominais por variáveis categóricas ordinais
df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}, inplace=True)
df['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}, inplace=True)
df['sex'].replace({'MALE':0, 'FEMALE': 1}, inplace=True)

# ======================== Atributos Previsores e Alvo ========================

# Separar os atributos previsores
X = df.iloc[:,0:-1]

# Separar a Variável alvo
Y = df.iloc[:,-1]

# ============================= Árvore de Decisão =============================

# Criar o Modelo de Árvore de Decisão
model = DecisionTreeClassifier(criterion="entropy") 

# Treinar o Modelo de Árvore de Decisão
model.fit(X, Y)

# Imprimir as regras que foram geradas
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(50,50))
tree.plot_tree(model, feature_names=X.columns, class_names={0:"MALE",1:"FEMALE"},filled=False)
plt.show()

# =============================================================================
# =============================== Segunda Etapa ===============================
# =============================================================================

filtro1 = df[df["body_mass_g"] > 3712.5]

plt.scatter(filtro1['body_mass_g'],filtro1['sex'],color="green") 
plt.title("Sexo do Pinguim x Massa Corporal")
plt.xlabel("Massa Corporal do Pinguim")
plt.ylabel("Sexo do Pinguim")  
plt.grid()
plt.show()

filtro2 = filtro1[filtro1["bill_depth_mm"] <= 14.85]

plt.scatter(filtro2['bill_depth_mm'],filtro2['sex'],color="blue") 
plt.title("Sexo do Pinguim x Altura do Bico")
plt.xlabel("Altura do bico do Pinguim")
plt.ylabel("Sexo do Pinguim")  
plt.grid()
plt.show()

filtro3 = filtro2[filtro2["body_mass_g"] <= 5250.0]

plt.scatter(filtro3['body_mass_g'],filtro3['sex'],color="black") 
plt.title("Sexo do Pinguim x Massa Corporal")
plt.xlabel("Massa Corporal do Pinguim")
plt.ylabel("Sexo do Pinguim")  
plt.grid()
plt.show()

filtro4 = filtro2[filtro2["body_mass_g"] > 5250.0]

plt.scatter(filtro4['body_mass_g'],filtro4['sex'],color="red") 
plt.title("Sexo do Pinguim x Massa Corporal")
plt.xlabel("Massa Corporal do Pinguim")
plt.ylabel("Sexo do Pinguim")  
plt.grid()
plt.show()

# =============================================================================
# ============================== Terceira Etapa ===============================
# =============================================================================
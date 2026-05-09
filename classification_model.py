import pandas as pd

# 1: Carregar dados
# Não há valores faltantes
df_transacoes = pd.read_csv("./dataset/transacoes_fraude.csv")
print(df_transacoes.info())
print(df_transacoes.describe())
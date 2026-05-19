import joblib
import matplotlib.pyplot as plt
import pandas as pd
from optuna import create_study
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1: Carregar dados
# Não há valores faltantes.
# Coluna id do cliente é inútil para previsão.
print("Carregar dados:")
df_transacoes = pd.read_csv("./dataset/transacoes_fraude.csv")

print(df_transacoes.info())
print(df_transacoes.describe())

df_transacoes.columns = [
  'client_id',
  'tipo',
  'valor',
  'saldo_antes',
  'saldo_depois',
  'data',
  'classe'
]
df_transacoes.drop(columns=['client_id'], inplace=True)

print("Após modificações:")
print(df_transacoes.head(5))

# 2: Exploratory Data Analysis (EDA)
print("\nAnálise Exploratória dos Dados (EDA):")
df_transacoes_eda = df_transacoes.copy()
df_transacoes_eda["classe_str"] = df_transacoes_eda.classe.map({
  0: "Não Fraude",
  1: 'Fraude'
})
print("Tipos de transações:")
print(df_transacoes_eda.tipo.unique())

# Ver outliers
sns.boxplot(data=df_transacoes_eda, y="valor")
plt.ylabel("Valor da Transação (BRL)")
plt.savefig("./dataviz/valor-boxplot.png")
plt.close()

sns.boxplot(data=df_transacoes_eda, y='saldo_antes')
plt.ylabel("Saldo antes da transação (BRL)")
plt.savefig("./dataviz/saldo-antes-boxplot.png")
plt.close()

sns.boxplot(data=df_transacoes_eda, y='saldo_depois')
plt.ylabel("Saldo depois da transação (BRL)")
plt.savefig("./dataviz/saldo-depois-boxplot.png")
plt.close()

# Tipo de transação não afeta fortemente a distribuição dos dados.
crosstab_classe_tipo = pd.crosstab(
  df_transacoes_eda.classe_str,
  df_transacoes_eda.tipo,
)
print(crosstab_classe_tipo)
sns.heatmap(
  crosstab_classe_tipo,
  annot=True,
  fmt='d',
  cmap='Purples',
  linewidths=.5,
  linecolor="black",
)
plt.ylabel("Classe")
plt.xlabel("Tipo de Transação")
plt.savefig("./dataviz/tipo-x-classe-heatmap.png")
plt.close()

# Valor da transação + Span
# Valor não aparenta influenciar
sns.boxplot(data=df_transacoes_eda, x='classe_str', y='valor')
plt.xlabel("Classe")
plt.ylabel("Valor")
plt.savefig("./dataviz/valor-x-classe-boxplot.png")
plt.close()

# Saldo depois da transação 
# Também não aparenta influenciar
sns.boxplot(data=df_transacoes_eda, x="classe_str", y="saldo_depois")
plt.xlabel("Classe")
plt.ylabel("Saldo após transação")
plt.savefig("./dataviz/saldo-depois-x-classe-boxplot.png")
plt.close()

# Saldo antes da transação
# Também aparenta não influenciar
sns.boxplot(data=df_transacoes_eda, x="classe_str", y="saldo_antes")
plt.xlabel("Classe")
plt.ylabel("Saldo antes da transação")
plt.savefig("./dataviz/saldo-antes-x-classe-boxplot.png")
plt.close()

# Análisar pela hora:
# madrugada = 0-6h, manhã = 6-12h, tarde=12-18h, noite=18-24
# As fraudes acontecem com maior volume na madrugada
df_transacoes_eda["datetime"] = pd.to_datetime(df_transacoes_eda.data)

df_transacoes_eda['periodo'] = df_transacoes_eda['datetime'].dt.hour.apply(
  lambda h:
    'Manhã' if 6 <= h < 12 else
    'Tarde' if 12 <= h < 18 else
    'Noite' if 18 <= h < 24 else
    'Madrugada'
)

crosstab_classe_periodo = pd.crosstab(
  df_transacoes_eda["classe_str"],
  df_transacoes_eda["periodo"]
)
sns.heatmap(
  crosstab_classe_periodo,
  cmap="Purples",
  annot=True,
  fmt="d"
)
plt.xlabel("Período do Dia")
plt.ylabel("Classe")
plt.savefig("./dataviz/periodo-x-classe-heatmap.png")
plt.close()

df_numeric = df_transacoes_eda.select_dtypes(include=['number'])
sns.heatmap(
    data=df_numeric.corr("pearson"),
    cmap='coolwarm',
    annot=True,
    fmt='.2f'
)
plt.savefig("./dataviz/pearson-correlation-heatmap.png")
plt.close()

# Analise específica de fraude
# A comparação mostra que os dados colidem muito 
# | e não são explicativos suficientemente
df_transacoes_eda_fraude = df_transacoes_eda[(df_transacoes_eda.classe_str == 'Fraude')]
print("Dados sem fraude:")
print(df_transacoes_eda[(df_transacoes_eda.classe_str == "Não Fraude")].describe())
print("Dados com fraude:")
print(df_transacoes_eda_fraude.describe())

# 3: Treinamento do modelo
print("\nTreinamento do Modelo:")

df_transacoes['data'] = pd.to_datetime(df_transacoes['data'])
df_transacoes['hora'] = df_transacoes['data'].dt.hour
df_transacoes.drop(columns=['data'], inplace=True)

X = df_transacoes.drop(columns=['classe'])
y = df_transacoes['classe']

categorical_features = ['tipo']
numeric_features = [
    'valor',
    'saldo_antes',
    'saldo_depois',
    'hora'
]

categorical_transformer = Pipeline(steps=[
  ('encoder', OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
  transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', 'passthrough', numeric_features)
  ]
)

model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('classifier', DecisionTreeClassifier(
    class_weight='balanced',
    random_state=51)
  )
])

cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=51)
metrics_result = cross_validate(
  model,
  X,
  y,
  cv=cv_folds,
  scoring=['accuracy'],
  return_estimator=True
)

print(metrics_result)
media_acuracia = metrics_result['test_accuracy'].mean()
print(f"Média de acurácia: {media_acuracia:.4f}")

y_pred = cross_val_predict(model, X, y, cv=cv_folds)
classification_report_str = classification_report(y_true=y, y_pred=y_pred)
print(f"Relatório de Classificação:\n{classification_report_str}")

model_confusion_matrix = confusion_matrix(
  y_true=y,
  y_pred=y_pred,
)

sns.heatmap(
  model_confusion_matrix,
  annot=True,
  fmt='d',
  cmap='Purples',
  xticklabels=['Não Fraude', 'Fraude'],
  yticklabels=['Não Fraude', 'Fraude']
)
plt.xlabel("Predição")
plt.ylabel("Valor Real")
plt.savefig("./dataviz/matriz-de-confusao.png")
plt.close()

# 4: Tuning
def decisiontree_optuna(trial):
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_depth = trial.suggest_int('max_depth', 2, 8)

    model.set_params(classifier__min_samples_leaf=min_samples_leaf)
    model.set_params(classifier__max_depth=max_depth)

    scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")

    return scores.mean()

estudo_decision_tree = create_study(direction='maximize')
estudo_decision_tree.optimize(decisiontree_optuna, n_trials=100)
print(f"Melhor acurácia: {estudo_decision_tree.best_value}")
print(f"Melhores parâmetros: {estudo_decision_tree.best_params}")

# 5: Printar árvore
preprocessor.fit(X)

X_transformado = preprocessor.transform(X)

decision_tree = DecisionTreeClassifier(
    min_samples_leaf=estudo_decision_tree.best_params['min_samples_leaf'],
    max_depth=estudo_decision_tree.best_params['max_depth'],
    class_weight='balanced',
    random_state=51
)

decision_tree.fit(X_transformado, y)

ohe_features = preprocessor.named_transformers_['cat']\
    .named_steps['encoder']\
    .get_feature_names_out(categorical_features)

all_features = list(ohe_features) + numeric_features

plt.figure(figsize=(20,10))

plot_tree(
    decision_tree,
    feature_names=all_features,
    class_names=['Não Fraude', 'Fraude'],
    filled=True,
    fontsize=8
)

plt.savefig("./dataviz/decision-tree.png")
plt.close()

# 6: Modelo tunado
model_tunado = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        min_samples_leaf=estudo_decision_tree.best_params['min_samples_leaf'],
        max_depth=estudo_decision_tree.best_params['max_depth'],
        class_weight='balanced',
        random_state=51
    ))
])

model_tunado.fit(X, y)

# Salvar modelo
joblib.dump(model_tunado, "./classification_model.pkl")
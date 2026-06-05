import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# RF01 - Gerando o dataset diretamente no código - OK ------------------------------------

def gerar_dataset_vendas(n_registros=200, seed=42):
    """Gera um dataset sintético de vendas com dados intencionalmente sujos."""

    random.seed(seed)
    np.random.seed(seed)

    produtos = ["Notebook", "Smartphone", "Tablet", "Monitor", "Teclado", "Mouse", "Headset"]
    categorias = {"Notebook": "Computadores", "Smartphone": "Celulares", "Tablet": "Celulares",
                  "Monitor": "Computadores", "Teclado": "Periféricos", "Mouse": "Periféricos",
                  "Headset": "Periféricos"}

    regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
    clientes = [f"Cliente_{i:03d}" for i in range(1, 51)]

    data_inicio = datetime(2024, 1, 1)
    dados = []

    for i in range(n_registros):
        produto = random.choice(produtos)
        quantidade = random.randint(1, 10)
        preco_base = {"Notebook": 3500, "Smartphone": 2200, "Tablet": 1800,
                          "Monitor": 1200, "Teclado": 250, "Mouse": 120,
                          "Headset": 350}
        preco_base = preco_base[produto]
        preco = round(preco_base * random.uniform(0.85, 1.15), 2)
        data = data_inicio + timedelta(days=random.randint(0, 364))

        # Inserindo dados intencionalmente sujos para limpeza
        if random.random() < 0.05:
            quantidade = None  # valor nulo
        if random.random() < 0.04:
            preco = None  # valor nulo
        if random.random() < 0.03:
            produto = " " + produto  # espaço extra (string suja)

        dados.append({
            "id_venda": i + 1,
            "data_venda": data.strftime("%Y-%m-%d") if random.random() > 0.02 else "DATA INVÁLIDA",
            "cliente": random.choice(clientes),
            "produto": produto,
            "categoria": categorias.get(produto.strip(), "Outros"),
            "regiao": random.choice(regioes),
            "quantidade": quantidade,
            "preco_unitario": preco
        })
    return pd.DataFrame(dados)

# Gerar e salvar
df_bruto = gerar_dataset_vendas()
df_bruto.to_csv("vendas.csv", index=False)
print(f"Dataset gerado com {len(df_bruto)} registros.")
print(df_bruto.head())

# RF02 - Inspecionar e Descrever os Dados - OK ---------------------------------------

def inspecionar_dados(df):
  """Exibe informações básicas do DataFrame."""
  print("\n=== INSPEÇÃO INICIAL DO DATASET ===")
  print(f"Shape: {df.shape}")
  print(f"\nColunas: {list(df.columns)}")
  print(f"\nTipos de dados:\n{df.dtypes}")
  print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")
  print(f"\nPrimeiros registros:\n{df.head()}")
  print(f"\nEstatísticas descritivas:\n{df.describe()}")

# RF03 - Limpar e tratar os dados - OK -----------------------------------------------

import re

def limpar_dados(df):

  """
  Limpa e trata o DataFrame de vendas.
  Retorna o DataFrame limpo e um relatório de limpeza.
  """
  n_inicial = len(df)
  relatorio = {}

  # 1. Remover espaços extras em colunas de texto

  colunas_texto = df.select_dtypes(include="object").columns
  for col in colunas_texto:
      df[col] = df[col].str.strip()

  # 2. Converter data e remover datas inválidas

  df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
  n_datas_invalidas = df["data_venda"].isnull().sum()
  df = df.dropna(subset=["data_venda"])
  relatorio["datas_invalidas_removidas"] = n_datas_invalidas

  # 3. Remover linhas com quantidade ou preço nulos

  n_antes = len(df)
  df = df.dropna(subset=["quantidade", "preco_unitario"])
  relatorio["linhas_nulas_removidas"] = n_antes - len(df)

  # 4. Garantir tipos numéricos corretos

  df["quantidade"] = df["quantidade"].astype(int)
  df["preco_unitario"] = df["preco_unitario"].astype(float)

  n_final = len(df)
  relatorio["registros_iniciais"] = n_inicial
  relatorio["registros_finais"] = n_final
  relatorio["registros_removidos_total"] = n_inicial - n_final

  print("\n=== RELATÓRIO DE LIMPEZA ===")
  for chave, valor in relatorio.items():
      print(f" {chave}: {valor}")

  return df, relatorio



# RF04 - Criar Colunas Derivadas com Transformações - OK -----------------------------------

def criar_colunas_derivadas(df):

  """Cria colunas calculadas e derivadas a partir do dataset limpo."""

  # Receita total por linha de venda

  df["receita_total"] = df["quantidade"] * df["preco_unitario"]
  df["receita_total"] = df["receita_total"].astype(float)

  # Extração de componentes de data

  df["mes"] = df["data_venda"].dt.month
  df["mes_nome"] = df["data_venda"].dt.strftime("%B") # nome do mês
  df["trimestre"] = df["data_venda"].dt.quarter.apply(lambda q: f"Q{q}")
  df["ano"] = df["data_venda"].dt.year

  # Classificação da receita por item com numpy.select (transformação condicional vetorizada)

  condicoes = [
    df["receita_total"] < 500,
    (df["receita_total"] >= 500) & (df["receita_total"] < 5000),
    df["receita_total"] >= 5000
    ]

  classificacoes = ["Baixo Valor", "Médio Valor", "Alto Valor"]
  df["faixa_receita_item"] = np.select(condicoes, classificacoes, default="Não Classificado")

  print("\n=== COLUNAS DERIVADAS CRIADAS ===")
  print(df[["data_venda", "receita_total", "mes", "trimestre", "faixa_receita_item"]].head())
  return df



# RF05 - Calcular Métricas Agregadas (groupby) - OK

def calcular_metricas(df):

  """Calcula e retorna métricas agregadas do dataset."""

  metricas = {}

  # Receita por mês

  por_mes = df.groupby("mes").agg(receita_total=("receita_total", "sum"),
                                  quantidade=("quantidade", "sum"),
                                  n_vendas=("id_venda", "count")
                                  ).reset_index().sort_values("mes")
  metricas["por_mes"] = por_mes

  # Top 5 produtos por receita

  top_produtos = df.groupby("produto")["receita_total"].sum()\
  .sort_values(ascending=False).head(5).reset_index()
  metricas["top_produtos"] = top_produtos

  # Receita por categoria

  por_categoria = df.groupby("categoria")["receita_total"].sum().reset_index()
  metricas["por_categoria"] = por_categoria

  # Receita por região

  por_regiao = df.groupby("regiao").agg(receita_total=("receita_total", "sum"),
                                        media_ticket=("receita_total", "mean")
                                        ).reset_index().sort_values("receita_total", ascending=False)
  metricas["por_regiao"] = por_regiao

  # Exibição

  for nome, tabela in metricas.items():
      print(f"\n=== {nome.upper().replace('_', ' ')} ===")
      print(tabela.to_string(index=False))

  return metricas

# RF06 - Segmentar Clientes por Nível de Gasto - OK --------------------------

def segmentar_clientes(df):

  """Segmenta clientes pelo total gasto usando groupby e lambda."""

  clientes = df.groupby("cliente")["receita_total"].sum().round(2).reset_index()
  clientes.columns = ["cliente", "total_gasto"]

  # Classificação usando função lambda com condicionais

  clientes["segmento"] = clientes["total_gasto"].apply(
                                    lambda gasto: "Ouro" if gasto > 15000
                                    else ("Prata" if gasto >= 5000 else "Bronze")
                                    )

  clientes = clientes.sort_values("total_gasto", ascending=False)
  print("\n=== SEGMENTAÇÃO DE CLIENTES ===")
  print(clientes.head(10).to_string(index=False))
  print(f"\nDistribuição de segmentos:\n{clientes['segmento'].value_counts()}")
  return clientes

# RF07 - Calcular Estatísticas com Numpy - OK --------------------------------

def calcular_estatisticas_numpy(df):

  """Usa NumPy para calcular estatísticas sobre as receitas."""

  print("\n=== ESTATÍSTICAS COM NUMPY ===\n")

  receitas = df["receita_total"].to_numpy() # Converte para array NumPy

  media = np.mean(receitas)
  mediana = np.median(receitas)
  desvio_padrao = np.std(receitas)
  total = np.sum(receitas)
  p25 = np.percentile(receitas, 25)
  p75 = np.percentile(receitas, 75)

  print(f"	Receita média por venda: R$ {media:.2f}")
  print(f"	Receita mediana por venda: R$ {mediana:.2f}")
  print(f"	Desvio padrão: R$ {desvio_padrao:.2f}")
  print(f"	Receita total: R$ {total:.2f}")
  print(f"	Percentil 25 (Q1): R$ {p25:.2f}")
  print(f"	Percentil 75 (Q3): R$ {p75:.2f}")

  # Broadcasting: normalizar receitas entre 0 e 1

  receitas_normalizadas = (receitas - receitas.min()) / (receitas.max()  - receitas.min())
  print(f"\n Receitas normalizadas (primeiros 5): {receitas_normalizadas[:5].round(4)}")

  # Operação vetorizada: identificar vendas acima da média sem loop

  acima_da_media = receitas[receitas > media]
  print(f"\n Vendas acima da média: {len(acima_da_media)} de {len(receitas)}")

  return {
  "media": media, "mediana": mediana, "desvio_padrao": desvio_padrao, "total": total
  }

# RF08 - Criar Visualizações com Matplotlib e Seaborn - OK ------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import os

def gerar_visualizacoes(df, metricas, output_dir="outputs/graficos"):

  """Gera e exporta visualizações dos dados de vendas."""
  os.makedirs(output_dir, exist_ok=True)

  # Configurações visuais globais

  sns.set_theme(style="whitegrid", palette="muted")
  plt.rcParams["figure.figsize"] = (12, 6)
  plt.rcParams["axes.titlesize"] = 14
  plt.rcParams["axes.labelsize"] = 12

  # --- Gráfico 1: Receita por Mês (linha) ---

  fig, ax = plt.subplots()
  por_mes = metricas["por_mes"]
  ax.plot(por_mes["mes"], por_mes["receita_total"], marker="o", linewidth=2, color="#2196F3")
  ax.fill_between(por_mes["mes"], por_mes["receita_total"], alpha=0.15, color="#2196F3")
  ax.set_title("Receita Total por Mês (2024)")
  ax.set_xlabel("Mês")
  ax.set_ylabel("Receita Total (R$)")
  ax.set_xticks(range(1, 13))
  ax.set_xticklabels(["Jan","Fev","Mar","Abr","Mai","Jun","Jul",
                      "Ago","Set","Out","Nov","Dez"],
                     rotation=45)
  plt.tight_layout()
  caminho = os.path.join(output_dir, "vendas_por_mes.png")
  plt.savefig(caminho, dpi=150)
  plt.close()
  print(f" Gráfico exportado: {caminho}")

  # --- Gráfico 2: Top 5 Produtos (barras horizontais) ---

  fig, ax = plt.subplots()
  top = metricas["top_produtos"]
  sns.barplot(data=top, y="produto", x="receita_total", ax=ax, palette="Blues_d")
  ax.set_title("Top 5 Produtos por Receita Total")
  ax.set_xlabel("Receita Total (R$)")
  ax.set_ylabel("Produto")

  for container in ax.containers:
    ax.bar_label(container, fmt="R$ %.0f", padding=5)

  plt.tight_layout()
  caminho = os.path.join(output_dir, "top_produtos.png")
  plt.savefig(caminho, dpi=150)
  plt.close()
  print(f" Gráfico exportado: {caminho}")

  # --- Gráfico 3: Distribuição de Receita por Região (boxplot) ---

  fig, ax = plt.subplots()
  sns.boxplot(data=df, x="regiao", y="receita_total", ax=ax, palette="Set2")
  ax.set_title("Distribuição de Receita por Transação – Por Região")
  ax.set_xlabel("Região")
  ax.set_ylabel("Receita por Venda (R$)")
  plt.xticks(rotation=30)
  plt.tight_layout()
  caminho = os.path.join(output_dir, "distribuicao_regioes.png")
  plt.savefig(caminho, dpi=150)
  plt.close()
  print(f" Gráfico exportado: {caminho}")

# --- Gráfico NOVO ---

# --- Gráfico 4: Distribuição de Receita por Periférico (PIZZA) ---

  fig, ax = plt.subplots()
  por_categoria = metricas["por_categoria"]
  ax.pie(por_categoria["receita_total"], labels=por_categoria["categoria"], autopct="%1.1f%%", startangle=90)
  ax.set_title("Distribuição de Receita por Categoria")
  plt.tight_layout()
  caminho = os.path.join(output_dir, "distribuicao_categorias.png")
  plt.savefig(caminho, dpi=150)
  plt.close()
  print(f" Gráfico exportado: {caminho}")

  print("\n=== VISUALIZAÇÕES GERADAS COM SUCESSO ===")

# RF09 - Criar uma Classe para o Pipeline - OK ---------------------------

class AnalisadorDeVendas:

  """
  Classe responsável por encapsular o pipeline de análise de vendas.
  Mantém o estado do DataFrame e os resultados intermediários.

  """

  def __init__(self, caminho_arquivo):

    """Inicializa o analisador com o caminho do arquivo de dados."""

    self.caminho_arquivo = caminho_arquivo
    self.df_bruto = None
    self.df_limpo = None
    self.metricas = {}
    self.clientes = None
    self.relatorio_limpeza = {}

  def carregar(self):

    """Lê o arquivo CSV e armazena o DataFrame bruto."""

    self.df_bruto = pd.read_csv(self.caminho_arquivo)
    print(f"[AnalisadorDeVendas] Arquivo carregado: {self.caminho_arquivo}")
    print(f" Registros carregados: {len(self.df_bruto)}")
    return self

  def limpar(self):

    """Limpa os dados e armazena o DataFrame tratado."""
    self.df_limpo, self.relatorio_limpeza = limpar_dados(self.df_bruto.copy())
    return self

  def transformar(self):

    """Aplica transformações e cria colunas derivadas."""

    self.df_limpo = criar_colunas_derivadas(self.df_limpo)
    return self

  def analisar(self):

    """Calcula métricas e segmentações."""

    self.metricas = calcular_metricas(self.df_limpo)
    self.clientes = segmentar_clientes(self.df_limpo)
    calcular_estatisticas_numpy(self.df_limpo)
    return self

  def visualizar(self):

    """Gera e exporta os gráficos."""

    gerar_visualizacoes(self.df_limpo, self.metricas)
    return self

  def exportar_relatorio(self, caminho="outputs/relatorio_resumo.csv"):

    """Exporta o relatório de métricas por mês em CSV."""

    os.makedirs("outputs", exist_ok=True)
    self.metricas["por_mes"].to_csv(caminho, index=False)
    print(f"\n[AnalisadorDeVendas] Relatório exportado: {caminho}")
    return self

  def resumo(self):

    """Exibe um resumo executivo do pipeline."""

    print("\n" + "="*50)
    print("\tRESUMO EXECUTIVO – SALESINSIGHT PY")
    print("="*50)
    print(f" Arquivo analisado:\t{self.caminho_arquivo}")
    print(f" Registros brutos:  {self.relatorio_limpeza.get('registros_iniciais', 'N/A')}")
    print(f" Registros limpos:  {self.relatorio_limpeza.get('registros_finais', 'N/A')}")

    receita = self.df_limpo["receita_total"].sum() if self.df_limpo is not None else 0
    print(f" Receita total anual:\tR$ {receita:,.2f}")

    if self.clientes is not None:
      top = self.clientes.iloc[0]
      print(f" Cliente top:\t{top['cliente']} (R$    {top['total_gasto']:,.2f})")
    print("="*50)

# RF10 - Usar Herança - OK -----------------------------------------------------

class AnalisadorComProjecao(AnalisadorDeVendas):
  """
  Extensão do AnalisadorDeVendas com funcionalidades de projeção simples.
  Herda todos os métodos da classe pai e adiciona projeção de tendência.

  """

  def __init__(self, caminho_arquivo, meses_projecao=3):
    super().__init__(caminho_arquivo)
    self.meses_projecao = meses_projecao
    self.projecoes = []

  def projetar_tendencia(self):

    """
    Projeta a receita dos próximos meses com base na média móvel dos últimos 3 meses.
    Método simples sem machine learning – baseado em médias.

    """
    if not self.metricas or "por_mes" not in self.metricas:
      print("[AVISO] Rode .analisar() antes de projetar.")
      return self

    por_mes = self.metricas["por_mes"].sort_values("mes")
    receitas_historicas = por_mes["receita_total"].to_numpy()

    # Média móvel dos últimos 3 meses como base da projeção

    ultimos_3 = receitas_historicas[-3:]
    media_movel = np.mean(ultimos_3)
    tendencia = np.std(ultimos_3) * 0.1 # fator de crescimento simples


    ultimo_mes = int(por_mes["mes"].max())

    print("\n=== PROJEÇÃO DE TENDÊNCIA (Média Móvel Simples) ===")
    print(f" Base: média dos últimos 3 meses = R${media_movel:,.2f}")
    self.projecoes = []

    for i in range(1, self.meses_projecao + 1):
      mes_projetado = (ultimo_mes + i - 1) % 12 + 1
      receita_projetada = media_movel + (tendencia * i)
      self.projecoes.append({"mes": mes_projetado, "receita_projetada": round(receita_projetada, 2)})
      print(f" Mês {mes_projetado:02d} (projeção): R${receita_projetada:,.2f}")
    return self

  def exibir_projecao_detalhada(self):

    """Exibe as projeções calculadas."""

    if not self.projecoes:
      print("[AVISO] Nenhuma projeção disponível. Rode .projetar_tendencia() primeiro.")
      return

    print("\n=== DETALHAMENTO DAS PROJEÇÕES ===")
    for p in self.projecoes:
        print(f" Mês {p['mes']:02d}: R${p['receita_projetada']:,.2f}")

# RF11 - Usar Funções Lambda e Funções de Ordem Superior - OK ------------------------------

# Usaremos df_col_derivadas que é um DataFrame com o campo "receita_total" e "quantidade"
# Lambda em apply (transformação condicional de coluna)

df_limpo_df, _ = limpar_dados(df_bruto)
df_col_derivadas = criar_colunas_derivadas(df_limpo_df)

df_col_derivadas["desconto"] = df_col_derivadas["receita_total"].apply(lambda x: 0.10 if x > 10000 else 0.05)

# Vamos ver se funciona, transoforma em um dicionário a lista de produtos
lista_produtos = df_col_derivadas[["produto", "receita_total"]].to_dict(orient="records")

# Lambda em sorted para ordenar lista de dicionários
produtos_ordenados = sorted(lista_produtos, key=lambda p: p["receita_total"], reverse=True)

# Lambda como filtro rápido
vendas_alto_valor = df_col_derivadas[df_col_derivadas["receita_total"].apply(lambda x: x > 5000)]

def processar_coluna(df_target, coluna, funcao_transformacao):
  """
  Aplica uma função de transformação a uma coluna do DataFrame.
  Demonstra o uso de funções como argumentos (higher-order function / callback).
  """
  df_target[f"{coluna}_transformado"] = df_target[coluna].apply(funcao_transformacao)
  print(f" Coluna '{coluna}_transformado' criada com sucesso.")
  return df_target

# Uso da função com lambda como callback
df_col_derivadas = processar_coluna(df_col_derivadas, "receita_total", lambda x: round(x / 1000, 2))
df_col_derivadas = processar_coluna(df_col_derivadas, "quantidade", lambda x: "Alto" if x > 5 else "Baixo")

# RF12 - Ler e Escrever Arqruivos (CSV e JSON) - OK ------------------------------------------

import json

def exportar_resultados(metricas, clientes, stats_numpy):

  """Exporta resultados em CSV e JSON."""

  os.makedirs("outputs", exist_ok=True)

  # Exportar CSV com métricas por mês
  caminho_csv = "outputs/metricas_por_mes.csv"
  metricas["por_mes"].to_csv(caminho_csv, index=False, encoding="utf-8-sig")
  print(f" CSV exportado: {caminho_csv}")

  # Exportar segmentação de clientes em CSV
  caminho_clientes = "outputs/segmentacao_clientes.csv"
  clientes.to_csv(caminho_clientes, index=False, encoding="utf-8-sig")
  print(f" CSV exportado: {caminho_clientes}")

  # Exportar estatísticas gerais em JSON
  caminho_json = "outputs/estatisticas_gerais.json"
  stats_serializaveis = {k: round(float(v), 2) for k, v in stats_numpy.items()}

  with open(caminho_json, "w", encoding="utf-8") as f:
    json.dump(stats_serializaveis, f, indent=4, ensure_ascii=False)
  print(f" JSON exportado: {caminho_json}")

  # Ler e exibir o JSON exportado para confirmar
  with open(caminho_json, "r", encoding="utf-8") as f: dados_lidos = json.load(f)
  print(f"\n Conteúdo do JSON exportado:\n {json.dumps(dados_lidos, indent=2)}")

# import re

def limpar_strings_com_regex(df):
  """
  Usa expressões regulares para limpeza de colunas de texto.
  Exemplos: remover caracteres especiais, padronizar formatos.
   """

  # 1. Remover caracteres não alfanuméricos do nome do cliente (exceto underline e espaço)

  df["cliente_limpo"] = df["cliente"].apply(lambda s: re.sub(r"[^a-zA-Z0-9_ ]", "", str(s)).strip())

  # 2. Identificar registros com padrão de ID inválido (deve ser "Cliente_XXX")

  padrao_cliente = re.compile(r"^Cliente_\d{3}$")
  df["cliente_valido"] = df["cliente_limpo"].apply(lambda s: bool(padrao_cliente.match(s)))

  n_invalidos = (~df["cliente_valido"]).sum()

  print(f"\n=== LIMPEZA COM REGEX ===")
  print(f" Clientes com formato inválido encontrados: {n_invalidos}")
  print(f" Amostra de clientes limpos:{df['cliente_limpo'].head(5).tolist()}")

  return df

# RF14 - Executar o Pipeline Completo (Ponto de Entrada) ------------------------------------

def main():
  """
  Função principal: executa o pipeline completo do SalesInsight PY.
  """
  print("\n" + "="*60)
  print("	SALESINSIGHT PY – Pipeline de Análise de Dados de Vendas")
  print("="*60)

  # Etapa 0: Gerar dataset (se necessário)
  if not os.path.exists("vendas.csv"):
    print("\n[INFO] Gerando dataset sintético...")
    df_gerado = gerar_dataset_vendas(n_registros=200)
    df_gerado.to_csv("vendas.csv", index=False)

  # Etapa 1 a 6: Pipeline via classe com herança
  analisador = AnalisadorComProjecao("vendas.csv", meses_projecao=3)
  (analisador
  .carregar()
  .limpar()
  .transformar()
  .analisar()
  .projetar_tendencia()
  .visualizar()
  .exportar_relatorio()
  )

  # Etapa extra: limpeza com regex

  analisador.df_limpo = limpar_strings_com_regex(analisador.df_limpo)

  # Etapa extra: exportação JSON

  stats = calcular_estatisticas_numpy(analisador.df_limpo)
  exportar_resultados(analisador.metricas, analisador.clientes, stats)

  # Resumo final
  analisador.resumo()
  analisador.exibir_projecao_detalhada()
  print("\n[CONCLUÍDO] Pipeline finalizado com sucesso!")

if __name__	== "__main__":
  main()

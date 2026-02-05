# analisis_sueldos_vsco

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FuncFormatter
import os

# =================================================================
# 1. CONFIGURACIÓN INICIAL
# =================================================================

print("--- 1. CONFIGURACIÓN INICIAL ---")

# Nombres de archivos 
ARCHIVO_2025 = '2025.2 - Sysarmy CLEAN.csv'
ARCHIVO_2020 = 'Sysarmy-2020-2.csv'

# Verificación de archivos
if not os.path.exists(ARCHIVO_2025):
    print(f"⚠ ADVERTENCIA: No se encuentra el archivo '{ARCHIVO_2025}' en la carpeta actual.")
if not os.path.exists(ARCHIVO_2020):
    print(f"⚠ ADVERTENCIA: No se encuentra el archivo '{ARCHIVO_2020}' en la carpeta actual.")

# Configuración visual
SEED = 3
np.random.seed(SEED)
pd.set_option('display.float_format', '{:,.2f}'.format)
sns.set_style("whitegrid")
GENDER_PALETTE = {'Hombre': 'orange', 'Mujer': 'green', 'Otros': '#7f7f7f'}

# =================================================================
# 2. FUNCIONES
# =================================================================

def clean_gender(gender):
    if 'Mujer' in str(gender): return 'Mujer'
    elif 'Hombre' in str(gender): return 'Hombre'
    else: return 'Otros'

def calculate_ic_95(data):
    n = len(data)
    if n < 30:
        return pd.Series({'Media': data.mean(), 'Mediana': data.median(), 'IC Inf': np.nan, 'IC Sup': np.nan, 'N': n})
    media = data.mean()
    std_err = data.sem()
    ic = stats.t.interval(0.95, df=n-1, loc=media, scale=std_err)
    return pd.Series({'Media': media, 'Mediana': data.median(), 'IC Inf': ic[0], 'IC Sup': ic[1], 'N': n})

# --- FORMATO: Muestra el número completo con comas (Ej: 3,500,000) ---
def currency_formatter(x, pos):
    return f'{x:,.0f}'

formatter = FuncFormatter(currency_formatter)

# =================================================================
# 3. CARGA Y LIMPIEZA
# =================================================================

df_gender_2025 = None
df_gender_2020 = None

# --- 2025 ---
if os.path.exists(ARCHIVO_2025):
    try:
        print(f"Cargando {ARCHIVO_2025}...")
        # skiprows=9 es necesario por el formato específico de la encuesta Sysarmy 2025
        df_2025 = pd.read_csv(ARCHIVO_2025, skiprows=9, low_memory=False)
        
        df_2025.rename(columns={
            'ultimo_salario_mensual_o_retiro_bruto_en_pesos_argentinos': 'salary',
            'genero': 'gender',
            'seniority': 'seniority', # A veces cambia el nombre, revisar si falla
            'seniority,_sal': 'seniority', # Variación posible
            'maximo_nivel_de_estudios': 'studies',
            'anos_de_experiencia': 'years_exp'
        }, inplace=True)

        # Corrección si la columna de experiencia tiene otro nombre
        if 'years_exp' not in df_2025.columns:
             cols = [c for c in df_2025.columns if 'experiencia' in c]
             if cols: df_2025.rename(columns={cols[0]: 'years_exp'}, inplace=True)
        
        # Unificación de nombre seniority si vino con sufijo
        cols_seniority = [c for c in df_2025.columns if 'seniority' in c]
        if cols_seniority and 'seniority' not in df_2025.columns:
             df_2025.rename(columns={cols_seniority[0]: 'seniority'}, inplace=True)

        df_2025['salary'] = pd.to_numeric(df_2025['salary'], errors='coerce')
        df_red_2025 = df_2025.dropna(subset=['salary', 'gender']).copy()

        # Outliers 2025
        Q1, Q3 = df_red_2025['salary'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_red_2025 = df_red_2025[(df_red_2025['salary'] >= 50000) & (df_red_2025['salary'] <= (Q3 + 3*IQR))].copy()
        df_red_2025['gender_clean'] = df_red_2025['gender'].apply(clean_gender)
        df_gender_2025 = df_red_2025[df_red_2025['gender_clean'].isin(['Hombre', 'Mujer'])].copy()
        print("Dataset 2025 cargado correctamente.")
    except Exception as e:
        print(f"Error cargando 2025: {e}")

# --- 2020 ---
if os.path.exists(ARCHIVO_2020):
    try:
        print(f"Cargando {ARCHIVO_2020}...")
        df_2020 = pd.read_csv(ARCHIVO_2020, low_memory=False)
        df_2020.rename(columns={'salary_monthly_BRUTO': 'salary', 'profile_gender': 'gender'}, inplace=True)
        df_2020['salary'] = pd.to_numeric(df_2020['salary'], errors='coerce')
        df_red_2020 = df_2020.dropna(subset=['salary', 'gender']).copy()

        # Outliers 2020
        lim_sup = df_red_2020['salary'].quantile(0.995)
        df_red_2020 = df_red_2020[(df_red_2020['salary'] >= 20000) & (df_red_2020['salary'] <= lim_sup)].copy()
        df_red_2020['gender_clean'] = df_red_2020['gender'].apply(clean_gender)
        df_gender_2020 = df_red_2020[df_red_2020['gender_clean'].isin(['Hombre', 'Mujer'])].copy()
        print("Dataset 2020 cargado correctamente.")
    except Exception as e:
        print(f"Error cargando 2020: {e}")

# =================================================================
# 5. GRÁFICOS (Ejes con Cifras Completas)
# =================================================================
if df_gender_2025 is not None:
    print("\n--- GRÁFICOS SOLICITADOS (2025) ---")

    # 1. BOXPLOT
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df_gender_2025, x='salary', y='gender_clean', order=['Hombre', 'Mujer'], palette=GENDER_PALETTE)
    ax.xaxis.set_major_formatter(formatter)
    plt.title('1. Boxplot Salarios (2025)')
    plt.xlabel('Salario Bruto (ARS)')
    plt.xticks(rotation=45)
    plt.ylabel('Género')
    plt.tight_layout() # Ajusta para que no se corte
    plt.show()

    # 2. HISTOGRAMA
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data=df_gender_2025, x='salary', hue='gender_clean', element="step", stat="density", common_norm=False, palette=GENDER_PALETTE)
    ax.xaxis.set_major_formatter(formatter)
    plt.title('2. Histograma de Salarios (2025)')
    plt.xlabel('Salario Bruto (ARS)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. DENSIDAD (KDE)
    plt.figure(figsize=(12, 6))
    ax = sns.kdeplot(data=df_gender_2025, x='salary', hue='gender_clean', fill=True, alpha=0.4, palette=GENDER_PALETTE)
    ax.xaxis.set_major_formatter(formatter)
    plt.title('3. Densidad (KDE) (2025)')
    plt.xlabel('Salario Bruto (ARS)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. BARRAS (Mediana)
    meds = df_gender_2025.groupby('gender_clean')['salary'].median().reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=meds, x='gender_clean', y='salary', order=['Hombre', 'Mujer'], palette=GENDER_PALETTE)
    ax.yaxis.set_major_formatter(formatter)
    plt.title('4. Barras - Mediana Salarial (2025)')
    plt.ylabel('Mediana Salarial (ARS)')
    plt.xlabel('Género')
    plt.show()

    # 5. TORTA
    counts = df_red_2025['gender_clean'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=[GENDER_PALETTE.get(g, 'gray') for g in counts.index])
    plt.title('5. Torta - Composición de la Muestra')
    plt.show()

    # 6. DISPERSIÓN (Jitter)
    plt.figure(figsize=(12, 6))
    ax = sns.stripplot(data=df_gender_2025, x='gender_clean', y='salary', hue='gender_clean', palette=GENDER_PALETTE, alpha=0.3, jitter=0.2)
    ax.yaxis.set_major_formatter(formatter)
    plt.title('6. Dispersión (2025)')
    plt.ylabel('Salario Bruto (ARS)')
    plt.xlabel('Género')
    plt.show()

# =================================================================
# 6. COMPARACIÓN 2020 vs 2025
# =================================================================
if df_gender_2020 is not None and df_gender_2025 is not None:
    print("\n--- EVOLUCIÓN (2020-2025) ---")

    res_20 = df_gender_2020.groupby('gender_clean')['salary'].apply(calculate_ic_95).unstack()
    res_20['Year'] = '2020'
    res_25 = df_gender_2025.groupby('gender_clean')['salary'].apply(calculate_ic_95).unstack()
    res_25['Year'] = '2025'

    df_comp = pd.concat([res_20, res_25]).reset_index()

    plt.figure(figsize=(12, 7))
    ax = sns.pointplot(data=df_comp, x='Year', y='Mediana', hue='gender_clean', palette=GENDER_PALETTE, capsize=0.1)
    ax.yaxis.set_major_formatter(formatter)

    # Agregamos IC Manualmente
    def add_ic(df_row, x_idx, color_map):
        plt.vlines(x=x_idx, ymin=df_row['IC Inf'], ymax=df_row['IC Sup'], color=color_map[df_row['gender_clean']], linewidth=3, alpha=0.5)

    for i, year in enumerate(['2020', '2025']):
        rows = df_comp[df_comp['Year'] == year]
        for _, row in rows.iterrows():
            offset = -0.05 if row['gender_clean'] == 'Mujer' else 0.05
            plt.vlines(x=i+offset, ymin=row['IC Inf'], ymax=row['IC Sup'], color=GENDER_PALETTE[row['gender_clean']], linewidth=3)

    plt.title('7. Evolución Salarial e IC 95% (2020 vs 2025)')
    plt.ylabel('Mediana Salarial (ARS)')
    plt.show()

# =================================================================
# 7. ANÁLISIS MULTIVARIADO
# =================================================================
if df_gender_2025 is not None:
    print("\n--- ANÁLISIS CRUZADO ---")

    # 7.1 Seniority
    if 'seniority' in df_gender_2025.columns:
        df_seniority = df_gender_2025[df_gender_2025['seniority'].isin(['Junior', 'Senior'])].copy()

        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(data=df_seniority, x='seniority', y='salary', hue='gender_clean', palette=GENDER_PALETTE, order=['Junior', 'Senior'])
        ax.yaxis.set_major_formatter(formatter)
        plt.title('8. Salarios por Género y Seniority (Junior vs Senior)')
        plt.ylabel('Salario Bruto (ARS)')
        plt.xlabel('Nivel de Antigüedad')
        plt.show()

    # 7.2 Estudios
    if 'studies' in df_gender_2025.columns and 'seniority' in df_gender_2025.columns:
        df_studies = df_gender_2025[df_gender_2025['seniority'] == 'Senior'].copy()
        orden_jerarquico = ['Terciario', 'Universitario', 'Posgrado/Especialización', 'Maestría']
        df_studies = df_studies[df_studies['studies'].isin(orden_jerarquico)]

        plt.figure(figsize=(14, 7))
        ax = sns.boxplot(data=df_studies, x='studies', y='salary', hue='gender_clean', palette=GENDER_PALETTE, order=orden_jerarquico)
        ax.yaxis.set_major_formatter(formatter)
        plt.title('9. Salarios por Género y Estudios (Solo Seniors)')
        plt.ylabel('Salario Bruto (ARS)')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

    # 7.3 Regresión
    if 'years_exp' in df_gender_2025.columns:
        df_gender_2025['years_exp'] = pd.to_numeric(df_gender_2025['years_exp'], errors='coerce')
        df_exp = df_gender_2025.dropna(subset=['years_exp'])
        df_exp = df_exp[df_exp['years_exp'] <= 30]

        plt.figure(figsize=(12, 8))
        sns.regplot(data=df_exp[df_exp['gender_clean']=='Hombre'], x='years_exp', y='salary', scatter_kws={'alpha':0.1}, label='Hombre', color='orange')
        sns.regplot(data=df_exp[df_exp['gender_clean']=='Mujer'], x='years_exp', y='salary', scatter_kws={'alpha':0.1}, label='Mujer', color='green')

        ax = plt.gca()
        ax.yaxis.set_major_formatter(formatter)

        plt.title('10. Regresión: Impacto de los Años de Experiencia')
        plt.ylabel('Salario Bruto (ARS)')
        plt.xlabel('Años de Experiencia')
        plt.legend()
        plt.show()
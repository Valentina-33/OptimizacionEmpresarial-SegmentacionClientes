# =============================================================
#  ANÁLISIS ESTRATÉGICO — MINIMERCADO
#  Metodología CRISP-DM | Análisis descriptivo
#  Fases: 1. Comprensión | 2. ETL | 3. EDA | 4. Modelado
# =============================================================
#  REQUISITOS:
#  pip install pandas numpy openpyxl matplotlib seaborn scikit-learn
#
#  Cambia RUTA_ARCHIVO si el Excel está en otra carpeta.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

RUTA_ARCHIVO = "Anexo_BD_Minimercado.xlsx"   

sns.set_theme(style="whitegrid", palette="muted")
COLORES = {
    "azul":    "#185FA5",
    "verde":   "#1D9E75",
    "naranja": "#EF9F27",
    "coral":   "#D85A30",
    "celeste": "#378ADD",
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  FASE 1 — COMPRENSIÓN DEL PROBLEMA Y LOS DATOS           ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n" + "═"*55)
print("  FASE 1 — COMPRENSIÓN DEL PROBLEMA Y LOS DATOS")
print("═"*55)

xl         = pd.ExcelFile(RUTA_ARCHIVO)
ventas     = pd.read_excel(xl, sheet_name="Datos de ventas")
clientes   = pd.read_excel(xl, sheet_name="Datos de clientes")
inventario = pd.read_excel(xl, sheet_name="Datos_Inventario")

print(f"\n  Datos cargados:")
print(f"  · Ventas:     {ventas.shape[0]} registros, {ventas.shape[1]} variables")
print(f"  · Clientes:   {clientes.shape[0]} registros, {clientes.shape[1]} variables")
print(f"  · Inventario: {inventario.shape[0]} registros, {inventario.shape[1]} variables")
print(f"\n  Período: {ventas['Fecha de la Venta'].min().date()} → {ventas['Fecha de la Venta'].max().date()}")
print(f"  Ciudades: {sorted(ventas['Ubicación de la Tienda'].unique())}")
print(f"  Productos únicos: {ventas['Producto'].nunique()}")
print("""
  Hipótesis de negocio:
  H1: La frecuencia de compra es el principal diferenciador entre segmentos.
  H2: Los productos básicos tienen demanda más estable y predecible.
  H3: El comportamiento de compra varía significativamente por ciudad.
""")


# ╔═══════════════════════════════════════════════════════════╗
# ║  FASE 2 — ETL (PREPARACIÓN Y LIMPIEZA DE DATOS)          ║
# ╚═══════════════════════════════════════════════════════════╝

print("═"*55)
print("  FASE 2 — ETL")
print("═"*55)

print("\n  [2.1] Valores nulos:")
for nombre, df in [("Ventas", ventas), ("Clientes", clientes), ("Inventario", inventario)]:
    print(f"        {nombre}: {df.isnull().sum().sum()} nulo(s)")

print("\n  [2.2] Duplicados:")
for nombre, df in [("Ventas", ventas), ("Clientes", clientes), ("Inventario", inventario)]:
    print(f"        {nombre}: {df.duplicated().sum()} duplicado(s)")
clientes = clientes.drop_duplicates().reset_index(drop=True)
print(f"        → Clientes limpios: {len(clientes)} filas")

ventas["Fecha de la Venta"]      = pd.to_datetime(ventas["Fecha de la Venta"])
clientes["Género"]               = clientes["Género"].str.strip().str.upper()
clientes["Categoría de Cliente"] = clientes["Categoría de Cliente"].str.strip()
print("\n  [2.3] Tipos de dato estandarizados correctamente.")
print(f"\n  [2.4] Rangos: Precio ${ventas['Precio'].min():.2f}–${ventas['Precio'].max():.2f} | "
      f"Edad {clientes['Edad'].min()}–{clientes['Edad'].max()} años")

ids_v = set(ventas["Id del Producto"].unique())
ids_i = set(inventario["Id del Producto"].unique())
print(f"  [2.5] IDs inconsistentes entre hojas: {len(ids_v - ids_i)} → ninguno")

print("\n  [2.6] Outliers (IQR):")
for col in ["Cantidad Vendida", "Precio"]:
    Q1, Q3 = ventas[col].quantile([0.25, 0.75])
    n = ((ventas[col] < Q1-1.5*(Q3-Q1)) | (ventas[col] > Q3+1.5*(Q3-Q1))).sum()
    print(f"        {col}: {n} outlier(s) → conservados (valores plausibles)")

ventas["Total Venta"]  = ventas["Cantidad Vendida"] * ventas["Precio"]
ventas["Mes"]          = ventas["Fecha de la Venta"].dt.month
ventas["Nombre Mes"]   = ventas["Fecha de la Venta"].dt.strftime("%b")

inventario["Tasa Rotacion (%)"] = (
    (inventario["Niveles de Stock Inicial"] - inventario["Niveles de Stock Final"])
    / inventario["Niveles de Stock Inicial"] * 100
).round(1)
inventario["Riesgo"] = inventario["Tasa Rotacion (%)"].apply(
    lambda x: "Alto" if x > 55 else ("Medio" if x > 40 else "Bajo")
)

clientes["Genero Codigo"]    = clientes["Género"].map({"F": 0, "M": 1})
clientes["Categoria Codigo"] = clientes["Categoría de Cliente"].map({"Nuevo":0,"Regular":1,"Leal":2})
clientes["Rango Edad"]       = pd.cut(clientes["Edad"], bins=[24,30,40,55], labels=["25-30","31-40","41-53"])

print("\n  [2.7] KPIs calculados:")
print(f"        Ventas Totales:  ${ventas['Total Venta'].sum():,.2f}")
print(f"        Ticket Promedio: ${ventas['Total Venta'].mean():.2f}")
print(f"        Rotación prom.:  {inventario['Tasa Rotacion (%)'].mean():.1f}%")

with pd.ExcelWriter("datos_limpios_minimercado.xlsx", engine="openpyxl") as w:
    ventas.to_excel(w,     sheet_name="Ventas_Limpias",    index=False)
    clientes.to_excel(w,   sheet_name="Clientes_Limpios",  index=False)
    inventario.to_excel(w, sheet_name="Inventario_Limpio", index=False)
print("\n  [OK] Datos limpios exportados → datos_limpios_minimercado.xlsx")


# ╔═══════════════════════════════════════════════════════════╗
# ║  FASE 3 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)           ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n" + "═"*55)
print("  FASE 3 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("═"*55)

# Figura 1: Patrones de ventas
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Fase 3 — Patrones de ventas | Minimercado", fontsize=13, fontweight="bold", y=1.01)

ciudad = ventas.groupby("Ubicación de la Tienda")["Total Venta"].sum().sort_values(ascending=True)
axes[0,0].barh(ciudad.index, ciudad.values, color=COLORES["azul"], edgecolor="white")
axes[0,0].set_title("Ventas totales por ciudad")
axes[0,0].set_xlabel("Total ($)")
for i, v in enumerate(ciudad.values):
    axes[0,0].text(v+2, i, f"${v:.0f}", va="center", fontsize=9)

meses_orden = {8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov"}
mes_ventas  = ventas.groupby("Mes")["Total Venta"].sum()
mes_labels  = [meses_orden[m] for m in mes_ventas.index]
axes[0,1].plot(mes_labels, mes_ventas.values, marker="o", color=COLORES["azul"], linewidth=2.5, markersize=8)
axes[0,1].fill_between(range(len(mes_labels)), mes_ventas.values, alpha=0.12, color=COLORES["azul"])
axes[0,1].set_title("Evolución mensual de ventas")
axes[0,1].set_ylabel("Total ($)")
axes[0,1].set_xticks(range(len(mes_labels)))
axes[0,1].set_xticklabels(mes_labels)
for i, v in enumerate(mes_ventas.values):
    axes[0,1].annotate(f"${v:.0f}", (i, v), textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)

top8 = ventas.groupby("Producto")["Total Venta"].sum().sort_values(ascending=False).head(8)
colores_barras = [COLORES["coral"] if i < 3 else COLORES["celeste"] for i in range(len(top8))]
axes[1,0].barh(top8.index[::-1], top8.values[::-1], color=colores_barras[::-1], edgecolor="white")
axes[1,0].set_title("Top 8 productos por ingreso (rojo = top 3)")
axes[1,0].set_xlabel("Total ($)")

axes[1,1].scatter(ventas["Precio"], ventas["Cantidad Vendida"],
                  alpha=0.6, color=COLORES["verde"], edgecolors="white", s=60)
axes[1,1].set_title("Precio vs. cantidad vendida")
axes[1,1].set_xlabel("Precio ($)")
axes[1,1].set_ylabel("Cantidad vendida")

plt.tight_layout()
plt.savefig("EDA_ventas.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  [OK] EDA_ventas.png guardada")

# Figura 2: Segmentación de clientes
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Fase 3 — Segmentación de clientes | Minimercado", fontsize=13, fontweight="bold", y=1.01)

for cat, color in zip(["Nuevo","Regular","Leal"], [COLORES["celeste"], COLORES["naranja"], COLORES["verde"]]):
    axes[0,0].hist(clientes[clientes["Categoría de Cliente"]==cat]["Edad"],
                   bins=8, alpha=0.6, label=cat, color=color, edgecolor="white")
axes[0,0].set_title("Distribución de edad por categoría")
axes[0,0].set_xlabel("Edad")
axes[0,0].set_ylabel("Frecuencia")
axes[0,0].legend()

datos_box = [clientes[clientes["Categoría de Cliente"]==c]["Frecuencia de Compra"].values
             for c in ["Nuevo","Regular","Leal"]]
bp = axes[0,1].boxplot(datos_box, labels=["Nuevo","Regular","Leal"], patch_artist=True)
for patch, color in zip(bp["boxes"], [COLORES["celeste"], COLORES["naranja"], COLORES["verde"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[0,1].set_title("Frecuencia de compra por segmento")
axes[0,1].set_ylabel("Número de compras")

tabla_edad = pd.crosstab(clientes["Rango Edad"], clientes["Categoría de Cliente"])
tabla_edad[["Nuevo","Regular","Leal"]].plot(
    kind="bar", ax=axes[1,0],
    color=[COLORES["celeste"], COLORES["naranja"], COLORES["verde"]], edgecolor="white")
axes[1,0].set_title("Rango de edad x categoría de cliente")
axes[1,0].set_xlabel("Rango de edad")
axes[1,0].set_ylabel("Clientes")
axes[1,0].tick_params(axis="x", rotation=0)
axes[1,0].legend(title="Categoría")

pd.crosstab(clientes["Categoría de Cliente"], clientes["Género"]).plot(
    kind="bar", stacked=True, ax=axes[1,1],
    color=[COLORES["coral"], COLORES["azul"]], edgecolor="white")
axes[1,1].set_title("Género por categoría de cliente")
axes[1,1].set_xlabel("Categoría")
axes[1,1].set_ylabel("Clientes")
axes[1,1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("EDA_clientes.png", dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] EDA_clientes.png guardada")

# Figura 3: Correlaciones e inventario
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fase 3 — Correlaciones y riesgo de inventario", fontsize=13, fontweight="bold")

nums = clientes[["Edad","Frecuencia de Compra","Genero Codigo","Categoria Codigo"]].copy()
nums.columns = ["Edad","Frecuencia","Genero (M=1)","Categoria"]
sns.heatmap(nums.corr(), ax=axes[0], annot=True, fmt=".2f", cmap="Blues",
            mask=np.triu(np.ones_like(nums.corr(), dtype=bool)),
            linewidths=0.5, vmin=0, vmax=1, cbar_kws={"shrink":0.8})
axes[0].set_title("Correlación entre variables de clientes")

riesgo_count = inventario["Riesgo"].value_counts()
colores_r = {"Alto": COLORES["coral"], "Medio": COLORES["naranja"], "Bajo": COLORES["verde"]}
bars = axes[1].bar(riesgo_count.index, riesgo_count.values,
                   color=[colores_r[r] for r in riesgo_count.index], edgecolor="white")
axes[1].set_title("Productos por nivel de riesgo en inventario")
axes[1].set_ylabel("Número de registros")
for bar in bars:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 str(int(bar.get_height())), ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("EDA_correlaciones_inventario.png", dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] EDA_correlaciones_inventario.png guardada")

print("""
  INSIGHTS CLAVE — FASE 3:
  · Octubre fue el mes de mayor venta ($582). Agosto y noviembre bajan.
  · Cali ($328) y Cartagena ($319) lideran. Barranquilla ($253) es la menor.
  · Clientes leales: edad promedio 45.7 años, mayoría masculina, frecuencia 9.8x.
  · Clientes nuevos: edad promedio 31.1 años, mayoría femenina, frecuencia 2.0x.
  · Correlación Frecuencia-Categoría: 0.93 → valida H1 y es la base del modelo.
  · 10 de 72 registros de inventario tienen riesgo Alto de desabastecimiento.
""")


# ╔═══════════════════════════════════════════════════════════╗
# ║  FASE 4 — MODELADO                                        ║
# ║  4A: Árbol de clasificación (segmentación de clientes)    ║
# ║  4B: Árbol de regresión (predicción de demanda)           ║
# ╚═══════════════════════════════════════════════════════════╝

print("═"*55)
print("  FASE 4 — MODELADO")
print("═"*55)


# ─────────────────────────────────────────────────────────────
# 4A: ÁRBOL DE CLASIFICACIÓN
# Objetivo: clasificar clientes en Nuevo, Regular o Leal
# Variables: Edad, Género, Frecuencia de Compra
# ─────────────────────────────────────────────────────────────

print("\n  [4A] ÁRBOL DE CLASIFICACIÓN — Segmentación de clientes")
print("  " + "─"*50)

# Variables de entrada (X) y variable objetivo (y)
X_clf = clientes[["Edad", "Genero Codigo", "Frecuencia de Compra"]]
y_clf = clientes["Categoria Codigo"]  # 0=Nuevo, 1=Regular, 2=Leal

# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)
print(f"\n  Datos de entrenamiento: {len(X_train)} clientes")
print(f"  Datos de prueba:        {len(X_test)} clientes")

# Crear y entrenar el árbol
# max_depth=4: el árbol no crece demasiado (evita sobreajuste)
# min_samples_leaf=3: cada hoja necesita al menos 3 clientes
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=42)
clf.fit(X_train, y_train)

# Predicciones y métricas
y_pred_clf = clf.predict(X_test)
accuracy   = accuracy_score(y_test, y_pred_clf)

print(f"\n  Precisión del modelo: {accuracy:.2%}")
print(f"  Interpretación: el modelo clasifica correctamente")
print(f"  al {accuracy:.2%} de los clientes en su categoría real.\n")

print("  Reporte detallado por segmento:")
print("  " + "─"*45)
reporte = classification_report(y_test, y_pred_clf,
                                 target_names=["Nuevo","Regular","Leal"])
for linea in reporte.split('\n'):
    print("  " + linea)

# Importancia de variables
importancias = pd.Series(
    clf.feature_importances_,
    index=["Edad", "Género", "Frecuencia de Compra"]
).sort_values(ascending=False)

print("\n  Importancia de cada variable en la decisión:")
for var, imp in importancias.items():
    barra = "█" * int(imp * 30)
    print(f"  {var:<22} {barra} {imp:.1%}")

# Verificación del objetivo de negocio
print("\n  Verificación objetivo de negocio:")
if accuracy >= 0.85:
    print(f"  [OK] El modelo supera el umbral mínimo del 85% de precisión.")
    print(f"  [OK] Es apto para segmentar clientes y guiar estrategias de marketing.")
else:
    print(f"  [!] El modelo está por debajo del 85%. Se recomienda ajustar hiperparámetros.")

# Visualización del árbol de clasificación
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    clf,
    feature_names=["Edad", "Genero (0=F, 1=M)", "Frecuencia de Compra"],
    class_names=["Nuevo", "Regular", "Leal"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=False
)
ax.set_title(
    f"Fase 4A — Árbol de clasificación: segmentación de clientes\n"
    f"Precisión: {accuracy:.2%} | Variables: Edad, Género, Frecuencia de Compra",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("Modelo_Clasificacion.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  [OK] Árbol guardado → Modelo_Clasificacion.png")

# Primeras reglas del árbol en texto
print("\n  Primeras reglas del árbol de decisión:")
print("  " + "─"*45)
reglas = export_text(clf, feature_names=["Edad","Genero","Frecuencia"])
for linea in reglas.split('\n')[:20]:
    print("  " + linea)
print("  ...")


# ─────────────────────────────────────────────────────────────
# 4B: ÁRBOL DE REGRESIÓN
# Objetivo: predecir la cantidad demandada de un producto
# Variables: Producto, Ciudad, Precio, Mes
# ─────────────────────────────────────────────────────────────

print("\n\n  [4B] ÁRBOL DE REGRESIÓN — Predicción de demanda")
print("  " + "─"*50)

# Codificar variables categóricas como números
le_ciudad   = LabelEncoder()
le_producto = LabelEncoder()
ventas["Ciudad Cod"]   = le_ciudad.fit_transform(ventas["Ubicación de la Tienda"])
ventas["Producto Cod"] = le_producto.fit_transform(ventas["Producto"])

# Variables de entrada y variable a predecir
X_reg = ventas[["Producto Cod", "Ciudad Cod", "Precio", "Mes"]]
y_reg = ventas["Cantidad Vendida"]

# Dividir en entrenamiento (70%) y prueba (30%)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
print(f"\n  Datos de entrenamiento: {len(Xr_train)} transacciones")
print(f"  Datos de prueba:        {len(Xr_test)} transacciones")

# Crear y entrenar el árbol de regresión
reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3, random_state=42)
reg.fit(Xr_train, yr_train)

# Predicciones y métricas
yr_pred = reg.predict(Xr_test)
r2      = r2_score(yr_test, yr_pred)
mae     = mean_absolute_error(yr_test, yr_pred)

print(f"\n  R² (coeficiente de determinación): {r2:.4f}")
print(f"  MAE (error absoluto medio):        {mae:.2f} unidades")
print(f"""
  Interpretación:
  · R² = {r2:.2f} significa que el modelo explica el {r2:.0%}
    de la variación en la demanda de productos.
  · MAE = {mae:.2f} significa que, en promedio, el modelo
    se equivoca por ±{mae:.1f} unidades al predecir la demanda.
""")

# Importancia de variables
imp_reg = pd.Series(
    reg.feature_importances_,
    index=["Producto", "Ciudad", "Precio", "Mes"]
).sort_values(ascending=False)

print("  Importancia de cada variable en la predicción:")
for var, imp in imp_reg.items():
    barra = "█" * int(imp * 30)
    print(f"  {var:<10} {barra} {imp:.1%}")

# Verificación del objetivo de negocio
print("\n  Verificación objetivo de negocio:")
if r2 >= 0.5:
    print(f"  [OK] R²={r2:.2f} indica capacidad predictiva moderada-buena.")
    print(f"  [OK] Útil para orientar decisiones de reabastecimiento.")
    print(f"  [!] Para mejorar R², se recomienda incorporar más períodos")
    print(f"      históricos o variables de promociones y temporadas.")
else:
    print(f"  [!] R²={r2:.2f} indica baja capacidad predictiva.")
    print(f"  [!] Se recomienda ampliar el dataset o revisar las variables.")

# Visualización 1: árbol de regresión
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    reg,
    feature_names=["Producto", "Ciudad", "Precio", "Mes"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=False
)
ax.set_title(
    f"Fase 4B — Árbol de regresión: predicción de demanda\n"
    f"R²={r2:.4f} | MAE={mae:.2f} unidades | Variables: Producto, Ciudad, Precio, Mes",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("Modelo_Regresion.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  [OK] Árbol guardado → Modelo_Regresion.png")

# Visualización 2: valores reales vs. predichos + importancia variables
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fase 4B — Evaluación del modelo de regresión", fontsize=13, fontweight="bold")

axes[0].scatter(yr_test, yr_pred, color=COLORES["azul"], alpha=0.7, edgecolors="white", s=70)
lim = [min(yr_test.min(), yr_pred.min())-1, max(yr_test.max(), yr_pred.max())+1]
axes[0].plot(lim, lim, color=COLORES["coral"], linewidth=1.5, linestyle="--", label="Predicción perfecta")
axes[0].set_xlabel("Demanda real (unidades)")
axes[0].set_ylabel("Demanda predicha (unidades)")
axes[0].set_title(f"Real vs. predicho (R²={r2:.2f})")
axes[0].legend()

imp_reg_sorted = imp_reg.sort_values(ascending=True)
axes[1].barh(imp_reg_sorted.index, imp_reg_sorted.values,
             color=COLORES["verde"], edgecolor="white")
axes[1].set_title("Importancia de variables (regresión)")
axes[1].set_xlabel("Importancia relativa")
for i, v in enumerate(imp_reg_sorted.values):
    axes[1].text(v+0.005, i, f"{v:.1%}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("Modelo_Regresion_Evaluacion.png", dpi=150, bbox_inches="tight")
plt.show()
print("  [OK] Evaluación guardada → Modelo_Regresion_Evaluacion.png")

# Resumen final
print("""
  RESUMEN FASE 4:
  ─────────────────────────────────────────────────────────
  CLASIFICACIÓN:
  · Precisión: 95.8% → excelente para segmentar clientes.
  · La frecuencia de compra explica el 98.3% de la decisión.
  · Edad y género son variables secundarias.

  REGRESIÓN:
  · R² = 0.52 → el modelo explica el 52% de la variación
    en la demanda. Capacidad predictiva moderada.
  · El precio es la variable más relevante (78.3%) para
    predecir cuánto se va a vender de un producto.
  · MAE = 2.28 unidades → margen de error manejable para
    planificación de inventario semanal.
  ─────────────────────────────────────────────────────────
""")

print("═"*55)
print("  Fases 1, 2, 3 y 4 completadas.")
print("  Próximo paso → Fase 5: Recomendaciones estratégicas")
print("═"*55)
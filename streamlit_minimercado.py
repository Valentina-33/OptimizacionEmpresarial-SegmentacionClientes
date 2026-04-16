import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st



st.set_page_config(layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    color: black !important;
}
[data-testid="stMetricLabel"] {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Minimercado | Analítica y Segmentación",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "Anexo_BD_Minimercado.xlsx"
CLEAN_FILE = BASE_DIR / "datos_limpios_minimercado.xlsx"

IMG_PATHS = {
    "etl": BASE_DIR / "ETL_visualizaciones.png",
    "ventas": BASE_DIR / "EDA_ventas.png",
    "clientes": BASE_DIR / "EDA_clientes.png",
    "inventario": BASE_DIR / "EDA_correlaciones_inventario.png",
    "clasificacion": BASE_DIR / "Modelo_Clasificacion.png",
    "regresion": BASE_DIR / "Modelo_Regresion.png",
    "regresion_eval": BASE_DIR / "Modelo_Regresion_Evaluacion.png",
}

CITY_COORDS = {
    "Bogotá": {"lat": 4.7110, "lon": -74.0721},
    "Medellín": {"lat": 6.2442, "lon": -75.5812},
    "Cali": {"lat": 3.4516, "lon": -76.5320},
    "Barranquilla": {"lat": 10.9685, "lon": -74.7813},
    "Cartagena": {"lat": 10.3910, "lon": -75.4794},
}

MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

COLORES_SEG = {
    "Nuevo": "#3B82F6",
    "Regular": "#F59E0B",
    "Leal": "#10B981",
}

COLORES_RIESGO = {
    "Alto": "#EF4444",
    "Medio": "#F59E0B",
    "Bajo": "#10B981",
}


@st.cache_data
def cargar_datos():
    file_to_use = CLEAN_FILE if CLEAN_FILE.exists() else RAW_FILE

    if file_to_use == CLEAN_FILE:
        ventas = pd.read_excel(file_to_use, sheet_name="Ventas_Limpias")
        clientes = pd.read_excel(file_to_use, sheet_name="Clientes_Limpios")
        inventario = pd.read_excel(file_to_use, sheet_name="Inventario_Limpio")
    else:
        ventas = pd.read_excel(file_to_use, sheet_name="Datos de ventas")
        clientes = pd.read_excel(file_to_use, sheet_name="Datos de clientes")
        inventario = pd.read_excel(file_to_use, sheet_name="Datos_Inventario")

        clientes = clientes.drop_duplicates().reset_index(drop=True)
        ventas["Fecha de la Venta"] = pd.to_datetime(ventas["Fecha de la Venta"])
        clientes["Género"] = clientes["Género"].astype(str).str.strip().str.upper()
        clientes["Categoría de Cliente"] = clientes["Categoría de Cliente"].astype(str).str.strip()
        ventas["Total Venta"] = ventas["Cantidad Vendida"] * ventas["Precio"]
        ventas["Mes"] = ventas["Fecha de la Venta"].dt.month
        ventas["Nombre Mes"] = ventas["Mes"].map(MESES)
        inventario["Tasa Rotacion (%)"] = (
            (inventario["Niveles de Stock Inicial"] - inventario["Niveles de Stock Final"])
            / inventario["Niveles de Stock Inicial"] * 100
        ).round(1)
        inventario["Riesgo"] = inventario["Tasa Rotacion (%)"].apply(
            lambda x: "Alto" if x > 55 else ("Medio" if x > 40 else "Bajo")
        )

    ventas["Fecha de la Venta"] = pd.to_datetime(ventas["Fecha de la Venta"])
    if "Total Venta" not in ventas.columns:
        ventas["Total Venta"] = ventas["Cantidad Vendida"] * ventas["Precio"]
    if "Mes" not in ventas.columns:
        ventas["Mes"] = ventas["Fecha de la Venta"].dt.month
    if "Nombre Mes" not in ventas.columns:
        ventas["Nombre Mes"] = ventas["Mes"].map(MESES)
    if "Riesgo" not in inventario.columns:
        inventario["Tasa Rotacion (%)"] = (
            (inventario["Niveles de Stock Inicial"] - inventario["Niveles de Stock Final"])
            / inventario["Niveles de Stock Inicial"] * 100
        ).round(1)
        inventario["Riesgo"] = inventario["Tasa Rotacion (%)"].apply(
            lambda x: "Alto" if x > 55 else ("Medio" if x > 40 else "Bajo")
        )

    clientes["Género"] = clientes["Género"].astype(str).str.upper()
    clientes["Rango Edad"] = pd.cut(
        clientes["Edad"], bins=[24, 30, 40, 55], labels=["25-30", "31-40", "41-53"]
    )

    return ventas, clientes, inventario


ventas, clientes, inventario = cargar_datos()

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #f7f7f7;
        border: 1px solid #ececec;
        padding: 12px 16px;
        border-radius: 16px;
    }
    .insight-box {
        background: #f7fafc;
        border-left: 5px solid #2563eb;
        padding: 0.9rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🛒 Minimercado")
st.sidebar.caption("Dashboard interactivo del análisis en Python")

pagina = st.sidebar.radio(
    "Ir a",
    [
        "Resumen ejecutivo",
        "ETL y KPIs",
        "Ventas",
        "Clientes",
        "Inventario",
        "Modelos",
        "Recomendaciones",
        "Datos",
    ],
)

ciudades = ["Todas"] + sorted(ventas["Ubicación de la Tienda"].dropna().unique().tolist())
meses_disp = ["Todos"] + [MESES[m] for m in sorted(ventas["Mes"].dropna().unique())]
segmentos = ["Todos"] + sorted(clientes["Categoría de Cliente"].dropna().unique().tolist())

ciudad_sel = st.sidebar.selectbox("Ciudad", ciudades)
mes_sel = st.sidebar.selectbox("Mes", meses_disp)
segmento_sel = st.sidebar.selectbox("Segmento", segmentos)

ventas_f = ventas.copy()
clientes_f = clientes.copy()

if ciudad_sel != "Todas":
    ventas_f = ventas_f[ventas_f["Ubicación de la Tienda"] == ciudad_sel]

if mes_sel != "Todos":
    ventas_f = ventas_f[ventas_f["Nombre Mes"] == mes_sel]

if segmento_sel != "Todos":
    clientes_f = clientes_f[clientes_f["Categoría de Cliente"] == segmento_sel]

if ciudad_sel != "Todas":
    ids_ciudad = ventas_f["Id del Producto"].unique()
    clientes_f = clientes_f[clientes_f["Id del Producto"].isin(ids_ciudad)]
    inventario_f = inventario[inventario["Id del Producto"].isin(ids_ciudad)].copy()
else:
    inventario_f = inventario.copy()

ventas_totales = ventas_f["Total Venta"].sum()
unidades = int(ventas_f["Cantidad Vendida"].sum())
ticket_prom = ventas_f["Total Venta"].mean() if not ventas_f.empty else 0
rotacion_prom = inventario_f["Tasa Rotacion (%)"].mean() if not inventario_f.empty else 0

acc_clf = 0.9583
r2_reg = 0.5164
mae_reg = 2.28


# -------------------- RESUMEN --------------------
if pagina == "Resumen ejecutivo":
    st.title("📊 Optimización empresarial y segmentación de clientes")
    st.caption("Metodología CRISP-DM | Minimercado")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        c1.metric("Ventas totales", f"${ventas_totales:,.2f}")
    with c2:
        c2.metric("Unidades vendidas", f"{unidades}")
    with c3:
        c3.metric("Ticket promedio", f"${ticket_prom:,.2f}")
    with c4:
        c4.metric("Rotación promedio", f"{rotacion_prom:.1f}%")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        ventas_ciudad = (
            ventas_f.groupby("Ubicación de la Tienda", as_index=False)["Total Venta"]
            .sum()
            .sort_values("Total Venta", ascending=False)
        )
        fig = px.bar(
            ventas_ciudad,
            x="Total Venta",
            y="Ubicación de la Tienda",
            orientation="h",
            text="Total Venta",
            title="Ventas totales por ciudad",
        )
        fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig.update_layout(height=400, yaxis_title="", xaxis_title="Total ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        mensual = ventas_f.groupby(["Mes", "Nombre Mes"], as_index=False)["Total Venta"].sum().sort_values("Mes")
        fig = px.line(
            mensual,
            x="Nombre Mes",
            y="Total Venta",
            markers=True,
            title="Evolución mensual de ventas",
        )
        fig.update_traces(mode="lines+markers+text", text=[f"${v:.0f}" for v in mensual["Total Venta"]], textposition="top center")
        fig.update_layout(height=400, xaxis_title="Mes", yaxis_title="Total ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hallazgos principales")
    st.markdown(
        f"""
        <div class="insight-box"><b>Segmentación:</b> el modelo de clasificación alcanzó una precisión de <b>{acc_clf:.2%}</b>, así que sirve bastante bien para diferenciar clientes nuevos, regulares y leales.</div>
        <div class="insight-box"><b>Demanda:</b> el árbol de regresión obtuvo un <b>R² de {r2_reg:.2f}</b> y un <b>MAE de {mae_reg:.2f}</b> unidades, lo que lo vuelve útil como apoyo, aunque no como pronóstico perfecto.</div>
        <div class="insight-box"><b>Inventario:</b> la mayor parte de los registros está en riesgo <b>medio</b>, pero sí hay productos con riesgo <b>alto</b> que necesitan reabastecimiento más frecuente.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
        <style>
        .insight-box {
            background-color: #1f2937;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 12px;
            color: white !important;
            font-size: 15px;
            border-left: 5px solid #3b82f6;
        }
        </style>
        """, unsafe_allow_html=True)

    mapa_df = (
        ventas_f.groupby("Ubicación de la Tienda", as_index=False)["Total Venta"]
        .sum()
        .rename(columns={"Ubicación de la Tienda": "Ciudad"})
    )
    if not mapa_df.empty:
        mapa_df["lat"] = mapa_df["Ciudad"].map(lambda x: CITY_COORDS[x]["lat"])
        mapa_df["lon"] = mapa_df["Ciudad"].map(lambda x: CITY_COORDS[x]["lon"])
        fig_map = px.scatter_mapbox(
            mapa_df,
            lat="lat",
            lon="lon",
            size="Total Venta",
            color="Total Venta",
            hover_name="Ciudad",
            hover_data={"lat": False, "lon": False, "Total Venta": ':.2f'},
            zoom=4,
            height=450,
            title="Mapa interactivo de ventas por ciudad",
            mapbox_style="open-street-map",
        )
        st.plotly_chart(fig_map, use_container_width=True)


# -------------------- ETL --------------------
elif pagina == "ETL y KPIs":
    st.title("🧹 ETL y validación de datos")

    c1, c2, c3 = st.columns(3)
    c1.metric("Nulos en ventas", int(ventas.isnull().sum().sum()))
    c2.metric("Duplicados eliminados en clientes", 1)
    c3.metric("Productos únicos", int(ventas["Producto"].nunique()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KPIs del negocio")
        kpi_df = pd.DataFrame({
            "Indicador": ["Ventas Totales", "Ticket Promedio", "Unidades Vendidas", "Rotación Promedio"],
            "Valor": [
                f"${ventas_totales:,.2f}",
                f"${ticket_prom:,.2f}",
                f"{unidades}",
                f"{rotacion_prom:.1f}%",
            ],
            "Uso": [
                "Mide el ingreso generado",
                "Promedio gastado por transacción",
                "Volumen total de productos vendidos",
                "Ayuda a vigilar sobrestock o quiebre",
            ],
        })
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    with col2:
        precios = px.histogram(
            ventas_f,
            x="Precio",
            nbins=10,
            title="Distribución de precios",
        )
        precios.update_layout(height=350)
        st.plotly_chart(precios, use_container_width=True)

    if IMG_PATHS["etl"].exists():
        st.subheader("Visualización base del ETL")
        st.image(str(IMG_PATHS["etl"]), use_container_width=True)

    st.subheader("Validaciones útiles para explicar al profe")
    st.markdown(
        """
        - Se revisaron valores nulos, duplicados y tipos de dato.
        - Las fechas quedaron en formato correcto y se derivó el campo mes.
        - Se calculó la tasa de rotación para clasificar el riesgo de inventario.
        - Se conservó el outlier de 22 unidades porque es raro, pero sí plausible.
        """
    )


# -------------------- VENTAS --------------------
elif pagina == "Ventas":
    st.title("💰 Análisis de ventas")

    col1, col2 = st.columns(2)
    with col1:
        top_prod = (
            ventas_f.groupby("Producto", as_index=False)["Total Venta"]
            .sum()
            .sort_values("Total Venta", ascending=False)
            .head(8)
        )
        fig = px.bar(
            top_prod,
            x="Total Venta",
            y="Producto",
            orientation="h",
            text="Total Venta",
            title="Top productos por ingreso",
        )
        fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig.update_layout(height=450, xaxis_title="Ingreso ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            ventas_f,
            x="Precio",
            y="Cantidad Vendida",
            size="Total Venta",
            color="Ubicación de la Tienda",
            hover_name="Producto",
            title="Precio vs. cantidad vendida",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    mensual_ciudad = ventas_f.groupby(["Nombre Mes", "Ubicación de la Tienda"], as_index=False)["Total Venta"].sum()
    orden_meses = [m for m in ["Ago", "Sep", "Oct", "Nov"] if m in mensual_ciudad["Nombre Mes"].unique()]
    mensual_ciudad["Nombre Mes"] = pd.Categorical(mensual_ciudad["Nombre Mes"], categories=orden_meses, ordered=True)
    mensual_ciudad = mensual_ciudad.sort_values(["Nombre Mes", "Ubicación de la Tienda"])
    fig = px.bar(
        mensual_ciudad,
        x="Nombre Mes",
        y="Total Venta",
        color="Ubicación de la Tienda",
        barmode="group",
        title="Ventas por mes y ciudad",
    )
    fig.update_layout(height=420, xaxis_title="Mes", yaxis_title="Total ($)")
    st.plotly_chart(fig, use_container_width=True)

    if IMG_PATHS["ventas"].exists():
        with st.expander("Ver versión estática de la fase 3 de ventas"):
            st.image(str(IMG_PATHS["ventas"]), use_container_width=True)


# -------------------- CLIENTES --------------------
elif pagina == "Clientes":
    st.title("👥 Segmentación de clientes")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes analizados", int(len(clientes_f)))
    c2.metric("Frecuencia promedio", f"{clientes_f['Frecuencia de Compra'].mean():.2f}" if not clientes_f.empty else "0")
    c3.metric("Edad promedio", f"{clientes_f['Edad'].mean():.1f} años" if not clientes_f.empty else "0")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            clientes_f,
            x="Edad",
            color="Categoría de Cliente",
            barmode="overlay",
            nbins=12,
            color_discrete_map=COLORES_SEG,
            title="Distribución de edad por categoría",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            clientes_f,
            x="Categoría de Cliente",
            y="Frecuencia de Compra",
            color="Categoría de Cliente",
            color_discrete_map=COLORES_SEG,
            title="Frecuencia de compra por segmento",
        )
        fig.update_layout(height=420, xaxis_title="Segmento")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        rango_tab = (
            clientes_f.groupby(["Rango Edad", "Categoría de Cliente"], observed=False)
            .size()
            .reset_index(name="Clientes")
        )
        fig = px.bar(
            rango_tab,
            x="Rango Edad",
            y="Clientes",
            color="Categoría de Cliente",
            barmode="group",
            color_discrete_map=COLORES_SEG,
            title="Rango de edad por categoría",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        genero_tab = clientes_f.groupby(["Categoría de Cliente", "Género"], as_index=False).size()
        fig = px.bar(
            genero_tab,
            x="Categoría de Cliente",
            y="size",
            color="Género",
            barmode="stack",
            title="Género por categoría",
        )
        fig.update_layout(height=420, yaxis_title="Clientes")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resumen por segmento")
    resumen_seg = (
        clientes.groupby("Categoría de Cliente", as_index=False)
        .agg(
            Edad_promedio=("Edad", "mean"),
            Frecuencia_promedio=("Frecuencia de Compra", "mean"),
            Clientes=("Edad", "count"),
        )
        .sort_values("Frecuencia_promedio")
    )
    st.dataframe(resumen_seg.round(2), use_container_width=True, hide_index=True)

    if IMG_PATHS["clientes"].exists():
        with st.expander("Ver versión estática de la segmentación"):
            st.image(str(IMG_PATHS["clientes"]), use_container_width=True)


# -------------------- INVENTARIO --------------------
elif pagina == "Inventario":
    st.title("📦 Riesgo de inventario")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        riesgo_tab = inventario_f["Riesgo"].value_counts().rename_axis("Riesgo").reset_index(name="Registros")
        fig = px.bar(
            riesgo_tab,
            x="Riesgo",
            y="Registros",
            color="Riesgo",
            color_discrete_map=COLORES_RIESGO,
            text="Registros",
            title="Productos por nivel de riesgo",
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rotacion_id = inventario_f.groupby("Id del Producto", as_index=False)["Tasa Rotacion (%)"].mean().sort_values("Tasa Rotacion (%)", ascending=False).head(10)
        fig = px.bar(
            rotacion_id,
            x="Tasa Rotacion (%)",
            y=rotacion_id["Id del Producto"].astype(str),
            orientation="h",
            title="Top productos con mayor rotación",
        )
        fig.update_layout(height=420, yaxis_title="Id Producto")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla prioritaria para reabastecimiento")
    tabla_riesgo = (
        inventario_f.groupby(["Id del Producto", "Frecuencia de Reabastecimiento"], as_index=False)
        .agg(
            Rotacion_promedio=("Tasa Rotacion (%)", "mean"),
            Stock_inicial_prom=("Niveles de Stock Inicial", "mean"),
            Stock_final_prom=("Niveles de Stock Final", "mean"),
        )
        .sort_values("Rotacion_promedio", ascending=False)
    )
    st.dataframe(tabla_riesgo.round(2).head(12), use_container_width=True, hide_index=True)

    if IMG_PATHS["inventario"].exists():
        with st.expander("Ver versión estática de correlaciones e inventario"):
            st.image(str(IMG_PATHS["inventario"]), use_container_width=True)


# -------------------- MODELOS --------------------
elif pagina == "Modelos":
    st.title("🌳 Modelos predictivos")

    c1, c2, c3 = st.columns(3)
    c1.metric("Precisión clasificación", f"{acc_clf:.2%}")
    c2.metric("R² regresión", f"{r2_reg:.4f}")
    c3.metric("MAE regresión", f"{mae_reg:.2f} unidades")

    st.info(
        "La clasificación quedó fuerte para segmentación. La regresión sirve como apoyo para inventario, pero conviene presentarla como capacidad predictiva moderada y no como predicción exacta."
    )

    tab1, tab2, tab3 = st.tabs(["Clasificación", "Regresión", "Lectura rápida"])

    with tab1:
        st.markdown("**Variables usadas:** Edad, Género y Frecuencia de Compra")
        st.markdown("**Idea central:** la variable dominante es la frecuencia de compra.")
        if IMG_PATHS["clasificacion"].exists():
            st.image(str(IMG_PATHS["clasificacion"]), use_container_width=True)

    with tab2:
        st.markdown("**Variables usadas:** Producto, Ciudad, Precio y Mes")
        col1, col2 = st.columns(2)
        with col1:
            if IMG_PATHS["regresion"].exists():
                st.image(str(IMG_PATHS["regresion"]), use_container_width=True)
        with col2:
            if IMG_PATHS["regresion_eval"].exists():
                st.image(str(IMG_PATHS["regresion_eval"]), use_container_width=True)

    with tab3:
        st.markdown(
            """
            - **Clasificación:** el 95.83% indica que el árbol separa muy bien los segmentos.
            - **Regresión:** un **R² de 0.5164** significa que explica cerca del 52% de la variación de la demanda.
            - **Error absoluto medio:** en promedio se equivoca en unas **2.28 unidades**.
            - **Lectura de negocio:** bastante útil para orientar decisiones, pero todavía mejorable con más historial y más variables.
            """
        )


# -------------------- RECOMENDACIONES --------------------
elif pagina == "Recomendaciones":
    st.title("💡 Recomendaciones estratégicas")

    top_riesgo = (
        inventario.groupby("Id del Producto", as_index=False)["Tasa Rotacion (%)"]
        .mean()
        .sort_values("Tasa Rotacion (%)", ascending=False)
        .head(5)["Id del Producto"].astype(str).tolist()
    )

    lider_ciudad = ventas.groupby("Ubicación de la Tienda")["Total Venta"].sum().sort_values(ascending=False)
    ciudad_top = lider_ciudad.index[0]
    ciudad_baja = lider_ciudad.index[-1]
    brecha = lider_ciudad.iloc[0] - lider_ciudad.iloc[-1]

    st.markdown(
        f"""
        <div class="insight-box"><b>Inventario:</b> priorizar los productos {', '.join(top_riesgo)}. Son los que muestran mayor rotación promedio y merecen reabastecimiento más frecuente.</div>
        <div class="insight-box"><b>Marketing:</b> enfocar esfuerzos en clientes <b>regulares</b>, porque son el grupo con más potencial de pasar a leales.</div>
        <div class="insight-box"><b>Ciudades:</b> <b>{ciudad_top}</b> lidera en ventas, mientras <b>{ciudad_baja}</b> queda rezagada. La brecha es de aproximadamente <b>${brecha:,.2f}</b>.</div>
        """,
        unsafe_allow_html=True,
    )

    recs = pd.DataFrame(
        {
            "Frente": ["Inventario", "Segmento regular", "Segmento nuevo", "Segmento leal", "Cobertura por ciudad"],
            "Acción sugerida": [
                "Pasar productos críticos a reabastecimiento mensual y subir stock entre 15% y 20%.",
                "Aplicar programa de puntos o descuento progresivo para aumentar frecuencia de compra.",
                "Usar promociones digitales y combos de entrada para acelerar la recompra.",
                "Mantener beneficios exclusivos y promociones anticipadas para retención.",
                "Redistribuir mejor el stock de productos top en ciudades con menor desempeño.",
            ],
        }
    )
    st.dataframe(recs, use_container_width=True, hide_index=True)


# -------------------- DATOS --------------------
else:
    st.title("🗂️ Tablas de datos")
    tab1, tab2, tab3 = st.tabs(["Ventas", "Clientes", "Inventario"])

    with tab1:
        st.dataframe(ventas_f, use_container_width=True)
    with tab2:
        st.dataframe(clientes_f, use_container_width=True)
    with tab3:
        st.dataframe(inventario_f, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.caption("Hecho en Streamlit con filtros, KPIs y visualizaciones interactivas.")

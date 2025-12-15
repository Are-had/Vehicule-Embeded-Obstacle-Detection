import os
import pandas as pd
import streamlit as st
import pydeck as pdk

# ----------------------------
# Configuration Streamlit
# ----------------------------
st.set_page_config(
    page_title="Démo PFE - Obstacles routiers",
    layout="wide"
)

st.title("Démo – Visualisation des obstacles détectés")

DATA_PATH = "data/obstacles.csv"

# ----------------------------
# Chargement des données
# ----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Sécurisation des types
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # image_path peut être vide / NaN → string propre
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].fillna("").astype(str)
    else:
        df["image_path"] = ""

    # Nettoyage lignes invalides
    df = df.dropna(subset=["latitude", "longitude"])

    return df


if not os.path.exists(DATA_PATH):
    st.error("Le fichier data/obstacles.csv est introuvable. Lance d'abord :")
    st.code("python generate_demo_data.py")
    st.stop()

df = load_data(DATA_PATH)

# ----------------------------
# Filtres (sidebar)
# ----------------------------
st.sidebar.header("Filtres")

types = sorted(df["type_objet"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Type d'obstacle",
    types,
    default=types
)

min_vol = float(df["volume"].min())
max_vol = float(df["volume"].max())

vol_range = st.sidebar.slider(
    "Volume estimé (m³)",
    min_value=min_vol,
    max_value=max_vol,
    value=(min_vol, max_vol)
)

filtered_df = df[
    (df["type_objet"].isin(selected_types)) &
    (df["volume"] >= vol_range[0]) &
    (df["volume"] <= vol_range[1])
].copy()

# ----------------------------
# Layout principal
# ----------------------------
col_map, col_info = st.columns([2, 1], gap="large")

# ----------------------------
# Carte
# ----------------------------
with col_map:
    st.subheader("Carte des obstacles")

    if len(filtered_df) == 0:
        st.warning("Aucun obstacle avec ces filtres.")
    else:
        center_lat = float(filtered_df["latitude"].mean())
        center_lon = float(filtered_df["longitude"].mean())

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position="[longitude, latitude]",
            get_radius=40,
            get_fill_color=[255, 0, 0, 160],
            pickable=True,
            auto_highlight=True
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=12,
            pitch=0
        )

        tooltip = {
            "text": (
                "ID: {id}\n"
                "Type: {type_objet}\n"
                "Volume: {volume} m³\n"
                "Lat: {latitude}\n"
                "Lon: {longitude}"
            )
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip
            )
        )

# ----------------------------
# Détails obstacle
# ----------------------------
with col_info:
    st.subheader("Détails de l'obstacle")

    if len(filtered_df) == 0:
        st.stop()

    selected_id = st.selectbox(
        "Choisir un obstacle",
        filtered_df["id"].tolist()
    )

    row = filtered_df[filtered_df["id"] == selected_id].iloc[0]

    st.markdown(f"**Type :** {row['type_objet']}")
    st.markdown(f"**Volume estimé :** {row['volume']} m³")
    st.markdown(f"**Latitude :** {row['latitude']}")
    st.markdown(f"**Longitude :** {row['longitude']}")

    img_path = row["image_path"].strip()

    if img_path and os.path.exists(img_path):
        st.image(
            img_path,
            caption=f"Image associée à {selected_id}",
            use_container_width=True
        )
    else:
        st.info("Aucune image associée (OK pour la démo).")

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(
    "Démo PFE – Données simulées (CSV) → visualisation Streamlit. "
    "Prochaine étape : base SQL + API + application mobile Flutter."
)


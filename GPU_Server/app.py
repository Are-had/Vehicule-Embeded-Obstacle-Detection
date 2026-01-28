import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import os

st.set_page_config(page_title="Obstacle Detection Dashboard", layout="wide")

st.title("Road Obstacle Detection Results")

LOG_CSV = "data_results/inference_logs.csv"
PREDICTED_DIR = "data_results/predicted"
ORIGINAL_DIR = "data_results/original"

if not os.path.exists(LOG_CSV):
    st.error("No data available yet")
    st.stop()

df = pd.read_csv(LOG_CSV)

st.sidebar.header("Filters")
total_frames = len(df)
st.sidebar.metric("Total Frames Processed", total_frames)

obstacle_filter = st.sidebar.multiselect(
    "Filter by Object Type",
    options=["All"] + df['objects'].unique().tolist(),
    default=["All"]
)

if "All" not in obstacle_filter and obstacle_filter:
    df = df[df['objects'].isin(obstacle_filter)]

st.header("Map View - Detected Obstacles")

if len(df) > 0:
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    for idx, row in df.iterrows():
        if row['objects'] != "None":
            color = 'red'
            icon = 'exclamation-sign'
        else:
            color = 'green'
            icon = 'ok-sign'
        
        popup_html = f"""
        <b>Frame:</b> {row['frame_id']}<br>
        <b>Objects:</b> {row['objects']}<br>
        <b>Coordinates:</b> ({row['lat']:.6f}, {row['lon']:.6f})
        """
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    folium_static(m, width=1200, height=600)

st.header("Detection Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Statistics")
    total_obstacles = len(df[df['objects'] != "None"])
    st.metric("Frames with Obstacles", total_obstacles)
    st.metric("Clean Frames", total_frames - total_obstacles)

with col2:
    st.subheader("Objects Detected")
    objects_list = []
    for obj_str in df['objects']:
        if obj_str != "None":
            objects_list.extend(obj_str.split(", "))
    
    if objects_list:
        obj_counts = pd.Series(objects_list).value_counts()
        st.bar_chart(obj_counts)

st.header("Frame Gallery")

selected_frame = st.selectbox(
    "Select Frame to View",
    df['frame_id'].tolist()
)

if selected_frame:
    frame_data = df[df['frame_id'] == selected_frame].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        if os.path.exists(frame_data['original_img']):
            st.image(frame_data['original_img'], use_container_width=True)
        else:
            st.warning("Original image not found")
    
    with col2:
        st.subheader("Detected Objects")
        if os.path.exists(frame_data['predicted_img']):
            st.image(frame_data['predicted_img'], use_container_width=True)
        else:
            st.warning("Predicted image not found")
    
    st.info(f"Objects: {frame_data['objects']}")
    st.info(f"Location: ({frame_data['lat']:.6f}, {frame_data['lon']:.6f})")

st.header("Raw Data")
st.dataframe(df, use_container_width=True)
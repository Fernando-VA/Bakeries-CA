import streamlit as st

def show_header(text_title: str):
  # Layout: logo + title side by side
  col1, col2 = st.columns ([1, 6])
  with coll:
    st. image("Assets/UP_Logo.jpg", width=200)
    
  with col2:
    st. title(text_title)
    st.caption("Developed for: *Business Intelligence (Graduate Level)*")
    st.caption("Instructor: Edgar Avalos-Gauna (2025), Universidad Panamericana")

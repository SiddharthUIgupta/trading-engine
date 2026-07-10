import streamlit.components.v1 as components
import os

_component_func = components.declare_component(
    "custom_ui",
    path=os.path.dirname(os.path.abspath(__file__))
)

def render_custom_ui(**kwargs):
    return _component_func(**kwargs)

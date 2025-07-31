import matplotlib.pyplot as plt
import streamlit as st

def plot_equity_curve(dates, curves, labels):
    fig, ax = plt.subplots(figsize=(10,4))
    for curve, label in zip(curves, labels):
        ax.plot(dates, curve, label=label)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

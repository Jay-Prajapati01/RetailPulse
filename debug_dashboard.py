#!/usr/bin/env python3
"""Minimal Streamlit app to debug loading issue"""

import streamlit as st
import sys
import time

st.set_page_config(page_title="Debug Dashboard", layout="wide")

st.title("RetailPulse Dashboard Debug")
st.write("Testing data loading...")

# Start timer
start = time.time()

try:
    st.info("Step 1: Importing data loaders...")
    from src.dashboard.data_access import load_dashboard_summary
    st.success(f"✓ Imports done in {time.time() - start:.2f}s")
    
    st.info("Step 2: Loading dashboard summary...")
    summary = load_dashboard_summary()
    st.success(f"✓ Dashboard summary loaded in {time.time() - start:.2f}s")
    
    st.info("Step 3: Displaying summary...")
    st.json(summary)
    st.success(f"✓ All done in {time.time() - start:.2f}s")
    
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.error(traceback.format_exc())

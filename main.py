import streamlit as st
import pandas as pd
import numpy as np
import time

# 0. Load model da train
model = load_model("Duong dan den model")

# 1. Nhan file upload len
file = st.file_uploader("Chon file anh")

if not (file is None):
    # Xu ly file anh, chuan hoa, chuyen thanh tensor, blah blah
    tensor = process(file)
    # Dua vao model predict
    result = model.predict(tensor)
    #
    class_id = np.argmax(result)  # 0: Chó, 1: Mèo
    class_name = ["Chó", "Mèo"]
    st.write("Ket qua nha dien: ", class_name[class_id])


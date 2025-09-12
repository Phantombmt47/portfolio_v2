#!/usr/bin/env python
# coding: utf-8

# Import các thư viện cần thiết.
# vnstock đã được cập nhật với giao diện thống nhất, pandas 3.0 có hành vi Copy-on-Write mới,
# và streamlit đã loại bỏ một số hàm thử nghiệm.

from vnstock import Vnstock
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import numpy as np

# Sử dụng st.session_state để quản lý trạng thái widget hiện đại
# Thay thế các hàm thử nghiệm hoặc lỗi thời.
if "history_df" not in st.session_state:
    st.session_state["history_df"] = None

st.title("Phân tích Cổ phiếu Việt Nam")

# Chú ý: vnstock có thể bị lỗi 403 Forbidden trên các nền tảng đám mây như Google Colab và Kaggle
# do các nhà cung cấp dữ liệu chặn truy cập từ các địa chỉ IP này.
# Sử dụng trên máy cục bộ hoặc máy chủ proxy có thể giải quyết vấn đề.
try:
    # Bước 1: Khởi tạo đối tượng vnstock với giao diện hợp nhất mới
    stock_client = Vnstock().stock(symbol='VCI', source='VCI')

    # Bước 2: Tải dữ liệu lịch sử bằng cách sử dụng hàm quote.history()
    # Các tham số ngày tháng được chỉ định để lấy dữ liệu.
    history_df = stock_client.quote.history(start='2020-01-01', end='2024-05-25')

    # Lưu DataFrame vào session_state để duy trì trạng thái.
    st.session_state["history_df"] = history_df
    
    st.write("Dữ liệu lịch sử đã được tải thành công:")
    st.dataframe(st.session_state["history_df"])

    # Phân tích với pandas và scipy.
    # Bước 3: Đảm bảo dữ liệu được xử lý đúng cách cho pandas 3.0 và scipy
    # Tạo một bản sao tường minh để tuân thủ hành vi Copy-on-Write mới của pandas
    working_df = st.session_state["history_df"].copy()

    # Thao tác dữ liệu tuân thủ Copy-on-Write
    working_df.loc[:, 'daily_return'] = working_df['close'].pct_change()
    
    st.write("Phân tích lợi nhuận hàng ngày:")
    st.line_chart(working_df['daily_return'])

    # Bước 4: Tối ưu hóa với scipy.optimize
    # Lưu ý: Các hàm scipy.optimize.minimize yêu cầu mảng NumPy làm đầu vào.
    # Cần chuyển đổi rõ ràng DataFrame pandas thành mảng NumPy.
    
    # Hàm mẫu để tối thiểu hóa (ví dụ: hàm Rosenbrock)
    # Lưu ý: `scipy.optimize` mong đợi một mảng NumPy.
    def rosen(x):
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    # Khởi tạo một mảng NumPy mẫu
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    
    # Chạy tối ưu hóa bằng phương thức Nelder-Mead
    res = minimize(rosen, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    
    st.write("Kết quả tối ưu hóa bằng scipy.optimize:")
    st.write("Giá trị tối thiểu tìm thấy:", res.fun)
    st.write("Vị trí tối thiểu:", res.x)

except Exception as e:
    st.error(f"Có lỗi xảy ra: {e}")
    st.warning("Vui lòng kiểm tra lại kết nối mạng và các phụ thuộc. "
               "Lỗi '403 - Forbidden' hoặc 'ConnectTimeout' thường xảy ra trên các nền tảng đám mây.")

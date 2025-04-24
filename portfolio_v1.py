#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
from vnstock import Vnstock
import streamlit as st
import matplotlib.pyplot as plt

# ======== GIAO DIỆN STREAMLIT ========
st.set_page_config(page_title="Tối ưu danh mục cổ phiếu", layout="wide")
st.title("📈 Tối ưu Danh Mục Đầu Tư Cổ Phiếu")

# Nhập mã cổ phiếu
symbols_input = st.text_input("Nhập 2–6 mã cổ phiếu, cách nhau bằng dấu phẩy (VD: FPT, MWG, VNM)")
time_options = {"1 năm": 1, "3 năm": 3, "5 năm": 5, "10 năm": 10}
time_range = st.selectbox("Chọn khoảng thời gian phân tích:", options=list(time_options.keys()))

if st.button("Phân tích"):
    try:
        ma_cp = [ma.strip().upper() for ma in symbols_input.split(',') if ma.strip()]
        if len(ma_cp) < 2 or len(ma_cp) > 6:
            st.error("⚠️ Bạn phải nhập từ 2 đến 6 mã cổ phiếu.")
            st.stop()

        so_nam = time_options[time_range]
        ket_thuc = datetime.today().strftime('%Y-%m-%d')
        bat_dau = (datetime.today() - timedelta(days=365 * so_nam)).strftime('%Y-%m-%d')
        nguon = 'VCI'

        def lay_gia_dong_cua(ma):
            stock = Vnstock().stock(symbol=ma, source=nguon)
            df = stock.quote.history(symbol=ma, start=bat_dau, end=ket_thuc, interval='1D')
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df['close'].rename(ma)

        dulieu = pd.concat([lay_gia_dong_cua(ma) for ma in ma_cp], axis=1).dropna()
        log_return = np.log(dulieu / dulieu.shift(1)).dropna()
        loi_nhuan_tb = log_return.mean() * 252
        ma_tran_cov = log_return.cov() * 252

        def hieu_suat(weights):
            r = np.dot(weights, loi_nhuan_tb)
            v = np.sqrt(np.dot(weights.T, np.dot(ma_tran_cov, weights)))
            return r, v

        def toi_uu(chien_luoc='sharpe'):
            so_cp = len(ma_cp)
            khoi_tao = np.ones(so_cp) / so_cp
            gioi_han = tuple((0, 1) for _ in range(so_cp))
            rang_buoc = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            if chien_luoc == 'sharpe':
                muc_tieu = lambda w: -hieu_suat(w)[0] / hieu_suat(w)[1]
            elif chien_luoc == 'min_risk':
                muc_tieu = lambda w: hieu_suat(w)[1]
            elif chien_luoc == 'max_return':
                muc_tieu = lambda w: -hieu_suat(w)[0]
            return minimize(muc_tieu, khoi_tao, method='SLSQP', bounds=gioi_han, constraints=rang_buoc)

        opt_sharpe = toi_uu('sharpe')
        opt_risk = toi_uu('min_risk')
        opt_return = toi_uu('max_return')

        def trich_xuat_ket_qua(opt):
            w = opt.x
            r, v = hieu_suat(w)
            return [f"{i*100:.2f}%" for i in w], f"{r*100:.2f}%", f"{v*100:.2f}%"

        ty_trong_sharpe, exp_r_sharpe, risk_sharpe = trich_xuat_ket_qua(opt_sharpe)
        ty_trong_risk, exp_r_risk, risk_risk = trich_xuat_ket_qua(opt_risk)
        ty_trong_return, exp_r_return, risk_return = trich_xuat_ket_qua(opt_return)

        df_kq = pd.DataFrame({
            'Mã cổ phiếu': ma_cp,
            'Tối ưu Sharpe (%)': ty_trong_sharpe,
            'Tối ưu Rủi ro thấp (%)': ty_trong_risk,
            'Tối ưu Lợi nhuận cao (%)': ty_trong_return,
        })

        df_kq.loc[len(df_kq)] = ['Kỳ vọng lợi nhuận', exp_r_sharpe, exp_r_risk, exp_r_return]
        df_kq.loc[len(df_kq)] = ['Độ biến động (rủi ro)', risk_sharpe, risk_risk, risk_return]

        st.subheader("📊 BẢNG PHÂN BỔ VÀ HIỆU SUẤT DANH MỤC")
        st.dataframe(df_kq, use_container_width=True)

        # ======== VẼ BIỂU ĐỒ ========
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.25
        index = np.arange(len(ma_cp))

        ax.bar(index, [float(i.strip('%')) for i in ty_trong_sharpe], bar_width, label='Sharpe')
        ax.bar(index + bar_width, [float(i.strip('%')) for i in ty_trong_risk], bar_width, label='Rủi ro thấp')
        ax.bar(index + 2 * bar_width, [float(i.strip('%')) for i in ty_trong_return], bar_width, label='Lợi nhuận cao')

        ax.set_xlabel('Mã cổ phiếu')
        ax.set_ylabel('Tỷ trọng (%)')
        ax.set_title('Tỷ trọng phân bổ theo chiến lược')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(ma_cp)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")


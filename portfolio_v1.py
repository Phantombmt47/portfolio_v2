import pandas as pd
import numpy as np
from vnstock import Vnstock
from scipy.optimize import minimize
import plotly.express as px
import streamlit as st

# Bước 1: Tải dữ liệu lịch sử cho nhiều mã cổ phiếu
st.title("Tối ưu hóa Danh mục đầu tư dựa trên Tỷ lệ Sharpe")

stock_symbols_input = st.text_input("Nhập mã cổ phiếu (phân tách bằng dấu phẩy, ví dụ: TNG, STK, HHP):", value="TNG, STK, HHP")
stock_symbols = [symbol.strip().upper() for symbol in stock_symbols_input.split(',') if symbol.strip()]

start_date = st.date_input("Ngày bắt đầu", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("Ngày kết thúc", value=pd.to_datetime('today'))

historical_data = {}
vnstock_instance = Vnstock()

if stock_symbols:
    st.subheader("Đang tải Dữ liệu Lịch sử...")
    for symbol in stock_symbols:
        try:
            # Sử dụng TCBS làm nguồn dữ liệu ví dụ, có thể thay đổi nếu cần
            stock_client = vnstock_instance.stock(symbol=symbol, source='TCBS')
            history_df = stock_client.quote.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if not history_df.empty:
                # Đảm bảo cột 'time' là kiểu datetime và đặt làm chỉ mục để kết hợp
                if 'time' in history_df.columns:
                    history_df['time'] = pd.to_datetime(history_df['time'])
                    history_df = history_df.set_index('time')

                # Xử lý các chỉ mục thời gian trùng lặp bằng cách giữ lại mục cuối cùng
                if not history_df.index.is_unique:
                    history_df = history_df[~history_df.index.duplicated(keep='last')]


                historical_data[symbol] = history_df[['close']].copy() # Chỉ giữ lại giá đóng cửa
                st.write(f"Đã tải dữ liệu thành công cho {symbol}")
            else:
                st.warning(f"Không tìm thấy dữ liệu lịch sử cho {symbol}.")

        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu cho {symbol}: {e}")

    # Kết hợp tất cả các giá đóng cửa vào một DataFrame duy nhất
    if historical_data:
        combined_close_prices = pd.concat(historical_data.values(), axis=1, keys=historical_data.keys())
        st.subheader("Dữ liệu Giá đóng cửa Lịch sử Kết hợp:")
        st.dataframe(combined_close_prices.head())
    else:
        st.info("Không có dữ liệu lịch sử nào được tải thành công cho bất kỳ mã nào.")
        combined_close_prices = pd.DataFrame() # Đảm bảo combined_close_prices được định nghĩa


# Bước 2: Tính toán lợi nhuận
returns_df = pd.DataFrame()
if not combined_close_prices.empty:
    st.subheader("Đang tính toán Lợi nhuận Hàng ngày...")
    # Tính toán lợi nhuận hàng ngày cho DataFrame kết hợp
    returns_df = combined_close_prices.pct_change().dropna()
    st.write("Lợi nhuận Hàng ngày:")
    st.dataframe(returns_df.head())
else:
    st.info("Không có dữ liệu giá đóng cửa kết hợp để tính toán lợi nhuận.")

# Bước 3: Tính toán các chỉ số hiệu suất danh mục đầu tư
expected_returns = pd.Series(dtype='float64')
annualized_covariance_matrix = pd.DataFrame()

if not returns_df.empty:
    st.subheader("Đang tính toán các Chỉ số Danh mục Đầu tư...")
    # Tính toán lợi nhuận kỳ vọng (trung bình lợi nhuận hàng ngày)
    expected_returns = returns_df.mean()

    # Tính toán lợi nhuận kỳ vọng hàng năm (giả sử 252 ngày giao dịch)
    annualized_expected_returns = expected_returns * 252
    st.write("Lợi nhuận Kỳ vọng Hàng năm:")
    st.dataframe(annualized_expected_returns)

    # Tính toán độ biến động hàng năm (độ lệch chuẩn lợi nhuận hàng ngày)
    annualized_volatility = returns_df.std() * np.sqrt(252)
    st.write("Độ biến động Hàng năm:")
    st.dataframe(annualized_volatility)

    # Tính toán ma trận hiệp phương sai hàng năm
    annualized_covariance_matrix = returns_df.cov() * 252
    st.write("Ma trận Hiệp phương sai Hàng năm:")
    st.dataframe(annualized_covariance_matrix)

else:
    st.info("Không có dữ liệu lợi nhuận để tính toán các chỉ số danh mục đầu tư.")


# Bước 4: Định nghĩa hàm tối ưu hóa danh mục đầu tư
def calculate_portfolio_metrics(weights, expected_returns, covariance_matrix):
    """
    Tính toán lợi nhuận, độ biến động và Tỷ lệ Sharpe của danh mục đầu tư.

    Args:
        weights (np.ndarray): Mảng NumPy chứa trọng số cho từng tài sản.
        expected_returns (pd.Series): Series pandas chứa lợi nhuận kỳ vọng cho từng tài sản.
        covariance_matrix (pd.DataFrame): DataFrame pandas biểu thị ma trận hiệp phương sai của lợi nhuận tài sản.

    Returns:
        tuple: Một tuple chứa:
            - portfolio_return (float): Lợi nhuận danh mục đầu tư đã tính.
            - portfolio_volatility (float): Độ biến động danh mục đầu tư đã tính.
            - sharpe_ratio (float): Tỷ lệ Sharpe đã tính (giả sử lãi suất phi rủi ro là 0).
    """
    portfolio_return = np.sum(weights * expected_returns)
    # Đảm bảo covariance_matrix là mảng numpy cho np.dot nếu cần, nhưng ở đây là DataFrame
    # Chuyển đổi thành mảng numpy để nhân ma trận
    cov_matrix_np = covariance_matrix.values
    weights_np = np.array(weights)

    portfolio_volatility = np.sqrt(np.dot(weights_np.T, np.dot(cov_matrix_np, weights_np)))

    # Tránh chia cho 0 nếu độ biến động bằng 0
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio

st.write("Đã định nghĩa hàm `calculate_portfolio_metrics`")


# Bước 5: Thực hiện tối ưu hóa danh mục đầu tư
def minimize_sharpe_ratio(weights, expected_returns, covariance_matrix):
    """
    Tính toán Tỷ lệ Sharpe âm để tối ưu hóa.

    Args:
        weights (np.ndarray): Mảng NumPy chứa trọng số cho từng tài sản.
        expected_returns (pd.Series): Series pandas chứa lợi nhuận kỳ vọng cho từng tài sản.
        covariance_matrix (pd.DataFrame): DataFrame pandas biểu thị ma trận hiệp phương sai của lợi nhuận tài sản.

    Returns:
        float: Tỷ lệ Sharpe âm.
    """
    # Đảm bảo trọng số là mảng numpy để nhất quán
    weights_np = np.array(weights)
    # Hàm calculate_portfolio_metrics đã xử lý pandas Series và DataFrames

    portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_metrics(weights_np, expected_returns, covariance_matrix)
    # Trả về Tỷ lệ Sharpe âm để tối thiểu hóa
    return -sharpe_ratio

if not annualized_expected_returns.empty and not annualized_covariance_matrix.empty:
    st.subheader("Đang thực hiện Tối ưu hóa Danh mục Đầu tư...")
    num_assets = len(annualized_expected_returns)

    # Dự đoán ban đầu: Trọng số bằng nhau
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    # Ràng buộc: Tổng trọng số = 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})

    # Giới hạn: Trọng số từ 0 đến 1
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Thực hiện tối ưu hóa
    try:
        optimization_result = minimize(minimize_sharpe_ratio, initial_weights, method='SLSQP',
                                       bounds=bounds, constraints=constraints,
                                       args=(annualized_expected_returns, annualized_covariance_matrix))

        # Bước 6: Hiển thị phân bổ danh mục đầu tư tối ưu
        if optimization_result.success:
            optimal_weights = optimization_result.x
            st.success("Tối ưu hóa thành công!")
            st.subheader("Trọng số Danh mục Đầu tư Tối ưu:")

            # Tạo DataFrame để hiển thị tốt hơn
            optimal_portfolio_df = pd.DataFrame({'Mã cổ phiếu': annualized_expected_returns.index,
                                                 'Trọng số Tối ưu': optimal_weights})
            # Định dạng trọng số dưới dạng phần trăm
            optimal_portfolio_df['Trọng số Tối ưu'] = optimal_portfolio_df['Trọng số Tối ưu'].apply(lambda x: f"{x:.2%}")

            st.dataframe(optimal_portfolio_df)

            # Tính toán và hiển thị các chỉ số danh mục đầu tư tối ưu
            optimal_return, optimal_volatility, optimal_sharpe_ratio = calculate_portfolio_metrics(optimization_result.x, annualized_expected_returns, annualized_covariance_matrix)
            st.write(f"Lợi nhuận Hàng năm của Danh mục Tối ưu: {optimal_return:.2%}")
            st.write(f"Độ biến động Hàng năm của Danh mục Tối ưu: {optimal_volatility:.2%}")
            st.write(f"Tỷ lệ Sharpe Tối ưu: {optimal_sharpe_ratio:.4f}") # Tỷ lệ Sharpe là tỷ lệ, không phải phần trăm

        else:
            st.error(f"Tối ưu hóa không thành công: {optimization_result.message}")

    except Exception as e:
        st.error(f"Lỗi trong quá trình tối ưu hóa: {e}")

else:
    st.info("Không đủ dữ liệu để thực hiện tối ưu hóa danh mục đầu tư.")


# Bước 7: (Tùy chọn) Trực quan hóa Ranh giới Hiệu quả
if not annualized_expected_returns.empty and not annualized_covariance_matrix.empty:
    st.subheader("Đang tạo Ranh giới Hiệu quả (Tùy chọn)...")
    num_portfolios = 5000  # Số lượng danh mục ngẫu nhiên cần tạo

    # Khởi tạo mảng để lưu trữ kết quả danh mục đầu tư
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    # Tạo các danh mục ngẫu nhiên
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Chuẩn hóa trọng số để tổng bằng 1

        all_weights[i, :] = weights

        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_metrics(weights, annualized_expected_returns, annualized_covariance_matrix)

        ret_arr[i] = portfolio_return
        vol_arr[i] = portfolio_volatility
        sharpe_arr[i] = sharpe_ratio

    # Lưu kết quả vào DataFrame
    portfolio_results_df = pd.DataFrame({'Độ biến động': vol_arr, 'Lợi nhuận': ret_arr, 'Tỷ lệ Sharpe': sharpe_arr})

    st.write("Đã tạo các danh mục ngẫu nhiên và tính toán các chỉ số.")

    # Tạo biểu đồ phân tán
    fig = px.scatter(portfolio_results_df, x='Độ biến động', y='Lợi nhuận', color='Tỷ lệ Sharpe',
                     title='Rủi ro so với Lợi nhuận của Danh mục (Các Danh mục Ngẫu nhiên)',
                     labels={'Độ biến động': 'Độ biến động Hàng năm', 'Lợi nhuận': 'Lợi nhuận Hàng năm'},
                     hover_data={'Độ biến động': ':.2%', 'Lợi nhuận': ':.2%', 'Tỷ lệ Sharpe': ':.4f'}) # Định dạng dữ liệu khi di chuột qua

    # Thêm đánh dấu cho danh mục tối ưu nếu tối ưu hóa thành công
    if 'optimal_return' in locals() and 'optimal_volatility' in locals():
        fig.add_scatter(x=[optimal_volatility], y=[optimal_return],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='star'),
                        name='Danh mục Tối ưu',
                        hoverinfo='text', # Sử dụng văn bản cho hover
                        text=f'Danh mục Tối ưu<br>Lợi nhuận: {optimal_return:.2%}<br>Độ biến động: {optimal_volatility:.2%}<br>Tỷ lệ Sharpe: {optimal_sharpe_ratio:.4f}')


    # Cập nhật bố cục để hiển thị phần trăm tốt hơn trên các trục
    fig.update_layout(
        xaxis=dict(tickformat=".2%"),
        yaxis=dict(tickformat=".2%")
    )


    # Hiển thị biểu đồ
    st.plotly_chart(fig)

else:
    st.info("Không đủ dữ liệu để trực quan hóa Ranh giới Hiệu quả.")

# Bước 8: Hoàn thành nhiệm vụ - Code được cung cấp ở trên, hướng dẫn dưới đây.
st.subheader("Hướng dẫn:")
st.markdown("""
1.  **Nhập Mã Cổ phiếu**: Nhập các mã cổ phiếu bạn muốn đưa vào danh mục đầu tư vào ô văn bản ở trên, phân tách bằng dấu phẩy (ví dụ: TNG, STK, HHP).
2.  **Chọn Khoảng Thời gian**: Chọn ngày bắt đầu và ngày kết thúc cho dữ liệu lịch sử.
3.  **Chạy Ứng dụng**: Nếu chạy dưới dạng ứng dụng Streamlit (`streamlit run your_script_name.py`), ứng dụng sẽ tự động cập nhật khi bạn thay đổi đầu vào. Trong sổ tay Colab, bạn thường lưu code này vào một tệp `.py` và chạy nó từ terminal.
4.  **Xem Kết quả**: Ứng dụng sẽ hiển thị dữ liệu lịch sử đã tải, lợi nhuận đã tính, các chỉ số danh mục đầu tư, trọng số danh mục tối ưu và biểu đồ Ranh giới Hiệu quả (tùy chọn).
""")

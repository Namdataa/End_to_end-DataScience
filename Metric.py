import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sqlalchemy import create_engine
from datetime import datetime
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import requests
from streamlit_extras.metric_cards import style_metric_cards
from scipy.stats import norm
import os
import urllib

# Setup Streamlit page configuration
st.set_page_config(page_title="Dashboard", page_icon="🌍", layout="wide")
st.header("PHÂN TÍCH DỰ BÁO & ĐỀ XUẤT")
st.sidebar.image("images.jpeg",caption="")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard","Analytics", "Forecasting"],
        icons=["house", "eye"],
        menu_icon="cast",
        default_index=0
    )

# Load CSS for custom styling
with open('style.css', encoding='utf-8') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


class DatabaseManager:
    def __init__(self, server, database, username, password):
        """
        Khởi tạo đối tượng DatabaseManager.

        Parameters:
            server (str): Tên server SQL Server.
            database (str): Tên database.
            username (str): Tên tài khoản đăng nhập.
            password (str): Mật khẩu tài khoản.
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.engine = None

    def connect(self):
        """
        Tạo kết nối đến SQL Server và khởi tạo SQLAlchemy engine.
        """
        try:
            params = urllib.parse.quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"Trusted_Connection=no;"
            )
            self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        except Exception as e:
            st.write(f"Không thể kết nối đến database. Lỗi: {e}")
class FilterManager:
    def __init__(self, data):
        """
        Khởi tạo FilterManager với dữ liệu đầu vào.
        
        Args:
            data (pd.DataFrame): Dữ liệu cần lọc.
        """
        self.data = data

    def render_filters(self):
        """
        Hiển thị các bộ lọc trên giao diện Streamlit sidebar và trả về các giá trị đã chọn.
        
        Returns:
            tuple: Các giá trị được chọn từ các bộ lọc.
        """
        # Filter options for brand
        brand = st.sidebar.multiselect(
            'SELECT BRAND',
            options=self.data['brand'].unique(),  # Không thêm tùy chọn 'All'
            default=self.data['brand'].unique()  # Mặc định chọn tất cả các brand
        )

        # Filter options for type of perfume
        type_perfume = st.sidebar.multiselect(
            'SELECT TYPE OF PERFUME',
            options=self.data['type'].unique(),
            default=self.data['type'].unique()
        )

        # Price range filter
        min_price, max_price = self.data['price'].min(), self.data['price'].max()
        selected_price_range = st.sidebar.slider(
            'SELECT PRICE RANGE',
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )

        # Other filters
        item_location = st.sidebar.multiselect(
            'SELECT ITEM LOCATION',
            options=self.data['itemLocation'].unique(),
            default=self.data['itemLocation'].unique()
        )

        weather = st.sidebar.multiselect(
            'SELECT WEATHER',
            options=self.data['weather'].unique(),
            default=self.data['weather'].unique()
        )

        address = st.sidebar.multiselect(
            'SELECT CUSTOMER ADDRESS',
            options=self.data['address'].unique(),
            default=self.data['address'].unique()
        )

        sex = st.sidebar.multiselect(
            'SELECT SEX',
            options=self.data['sex'].unique(),
            default=self.data['sex'].unique()
        )

        # Age range filter
        min_age, max_age = self.data['age'].min(), self.data['age'].max()
        selected_age_range = st.sidebar.slider(
            'SELECT AGE RANGE',
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

        # Datetime Filters
        st.sidebar.title('Datetime Filters')
        month_review_filter = st.sidebar.multiselect(
            'SELECT MONTH OF REVIEW',
            options=list(range(1, 13)),
            default=list(range(1, 13))
        )

        year_review_filter = st.sidebar.multiselect(
            'SELECT YEAR OF REVIEW',
            options=list(range(min(self.data['order_time']).year, max(self.data['order_time']).year + 1)),
            default=list(range(min(self.data['order_time']).year, max(self.data['order_time']).year + 1))
        )

        return (brand, type_perfume, selected_price_range, item_location, weather,
                address, sex, selected_age_range, month_review_filter, year_review_filter)
    def filter_forecasting(self):
        key_forecasting = None
        time_keyword = None
        location = None
        time_weather = None
        with st.sidebar.form(key="forecasting_form"):
            key_forecasting = st.text_input("Nhập từ khóa muốn dự báo lượng truy cập:")
            time_keyword = st.text_input("Nhập số tháng muốn dự báo lượng truy cập:")

        # Nút submit cho form
            submit_forecasting = st.form_submit_button(label="Enter")

    # Xử lý khi bấm nút "Enter" cho form dự báo lượng truy cập
        if submit_forecasting:
            if key_forecasting and time_keyword:
                st.write(f"Bạn đã nhập từ khóa muốn dự báo: {key_forecasting} và thời gian dự báo {time_keyword} tháng")
            else:
                st.warning("Vui lòng nhập giá trị!")

    # Form 2: Dự báo thời tiết
        with st.sidebar.form(key="weather_forecasting_form"):
            location = st.selectbox(
                'Chọn địa điểm muốn dự báo',
                options=filtered_data['address'].unique(),  # Các lựa chọn cho selectbox
                index=0  # Tùy chọn: bạn có thể thiết lập lựa chọn mặc định là phần tử đầu tiên
            )
            time_weather = st.text_input("Nhập thời gian muốn dự báo thời tiết:", value="90")

        # Nút "Enter Weather Time" để submit form
            submit_weather = st.form_submit_button(label="Enter")

    # Xử lý khi bấm nút "Enter" cho form dự báo thời tiết
        if submit_weather:
            if time_weather:
                st.write(f"Bạn đã nhập thời gian dự báo thời tiết: {time_weather}")
            else:
                st.warning("Vui lòng nhập giá trị!")

    # Tùy chỉnh CSS cho nút để thay đổi màu sắc nút
        st.sidebar.markdown(
            '<style>div.stButton > button:first-child {color: red;}</style>',
            unsafe_allow_html=True
        )
        return  key_forecasting,location,time_keyword,time_weather
class CustomerMetrics:
    def __init__(self, data):
        """
        Khởi tạo CustomerMetrics với dữ liệu đầu vào.

        Args:
            data (pd.DataFrame): Dữ liệu giao dịch.
        """
        self.data = data

    def calculate_customer_retention_rate(self, customer_id_col):
        """
        Tính tỷ lệ giữ chân khách hàng (Customer Retention Rate).

        Args:
            customer_id_col (str): Tên cột chứa ID khách hàng.

        Returns:
            float: Tỷ lệ giữ chân khách hàng (phần trăm).
        """
        initial_customers = self.data[customer_id_col].nunique()  # Số khách hàng ban đầu
        repeat_customers = self.data[customer_id_col].value_counts()[self.data[customer_id_col].value_counts() > 1].count()  # Số khách hàng quay lại
        retention_rate = (repeat_customers / initial_customers) * 100 if initial_customers > 0 else 0  # Tỷ lệ giữ chân
        return retention_rate

    def calculate_total_orders(self, order_id_col):
        """
        Tính tổng số lượng đơn hàng.

        Args:
            order_id_col (str): Tên cột chứa ID đơn hàng.

        Returns:
            int: Tổng số lượng đơn hàng.
        """
        total_orders = self.data[order_id_col].nunique()
        return total_orders

    def calculate_aov(self, total_col, order_id_col):
        """
        Tính giá trị đơn hàng trung bình (AOV).

        Args:
            total_col (str): Tên cột chứa tổng giá trị đơn hàng.
            order_id_col (str): Tên cột chứa ID đơn hàng.

        Returns:
            float: Giá trị đơn hàng trung bình (AOV).
        """
        total_revenue = self.data[total_col].sum()
        total_orders = self.data[order_id_col].nunique()
        aov = total_revenue / total_orders if total_orders > 0 else 0
        return aov

    def calculate_clv(self, customer_id_col, order_id_col, total_col, retention_rate=0.6, discount_rate=0.1):
        # Tính trung bình giá trị đơn hàng
        average_order_value = self.data.groupby(customer_id_col)[total_col].mean().mean()

        # Tính tần suất mua hàng (số đơn hàng trung bình mỗi khách hàng)
        purchase_frequency = self.data.groupby(customer_id_col)[order_id_col].nunique().mean()

        # Tính CLV
        clv = (average_order_value * purchase_frequency * retention_rate) / (1 + discount_rate - retention_rate)

        return clv

class Forecasting:
    
    def __init__(self, api_url, api_key_a, api_key_b, model_path=None):
        self.url = api_url
        self.api_key_a = api_key_a
        self.api_key_b = api_key_b
        self.model_path = model_path
        
    def get_API(self, key):
        querystring = {"keyword": f"{key}", "country": "us"}
        headers = {
            "x-rapidapi-key": self.api_key_a,
            "x-rapidapi-host": self.api_key_b
        }
        response = requests.get(self.url, headers=headers, params=querystring)
        data=response.json()
        df=None
        try:
            # Thử cấu trúc đầu tiên
            df = pd.DataFrame(data[0]['monthly_search_volumes'], columns=['month', 'searches', 'year'])
        except (KeyError, IndexError) as e:
            # Nếu gặp lỗi KeyError hoặc IndexError, thử cấu trúc thứ hai
            try:
              #  st.write("Đang ở chỗ có result")
                df = pd.DataFrame(data['result'][0]['monthly_search_volumes'], columns=['month', 'searches', 'year'])
               # st.write(df)
            except KeyError as e:
                st.write("Đã hết lượt API trong ngày hôm nay hoặc bạn có thể đăng kí ở tài khoản khác")
            except Exception as e:
                st.write("Lỗi khác:", e)
        except Exception as e:
            st.write("Lỗi không xác định:", e)
        if df is not None and not df.empty:
            df['Month'] = df['month'] + ' ' + df['year'].astype(str)
            result_df = df[['Month', 'searches']]
            result_df['Month'] = pd.to_datetime(result_df['Month'], format='%B %Y')
            result_df.set_index('Month', inplace=True)
            return result_df
        else:
            return None
    def forecast_with_sarima(self, result_df, n_pred_periods, key_forecasting):
        train_size = int(len(result_df) * 0.8)
        train = result_df['searches'][:train_size]
        test = result_df['searches'][train_size:]
        
        # Create SARIMA model
        model_sarima = auto_arima(result_df['searches'], seasonal=True, m=2, d=1, trace=True, 
                                  error_action='ignore', suppress_warnings=True, stepwise=True)

        # Predict future
        fitted, confint = model_sarima.predict(n_periods=n_pred_periods, return_conf_int=True)
        date = pd.date_range(start=result_df['searches'].index[-1], periods=n_pred_periods + 1, freq='MS')[1:]
        
        # Convert prediction to Series
        fitted_seri = pd.Series(fitted, index=date, name='Dự đoán lượt tìm kiếm')
        fitted_seri.index.name = "Tháng"

        # Calculate confidence intervals
        lower = confint[:, 0]
        upper = confint[:, 1]

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['searches'], label='Actual')
        plt.step(fitted_seri.index, fitted_seri, color='red', linestyle='--', label='Forecast', where='post')

        # Connect last actual data point to forecasted data
        last_actual = result_df['searches'].iloc[-1]
        first_forecast_date = fitted_seri.index[0]
        plt.plot([result_df.index[-1], first_forecast_date], [last_actual, fitted_seri.iloc[0]],
                 color='orange', linestyle='--', label='Connection')

        plt.ylim((0, max(result_df['searches'].max(), fitted_seri.max()) + 1000))
        plt.xlim((result_df.index[0], fitted_seri.index[-2]))
        plt.legend()
        plt.title(f'Dự báo lưu lượng tìm kiếm cho {key_forecasting}')
        plt.xlabel('Date')
        plt.ylabel('Search Count')
        plt.grid()
        plt.show()

        return fitted_seri,plt

    def forecast_weather(self, df, forecast_days,location_forecasting, time_step=30):
        if self.model_path is None:
            raise ValueError("Model path is not provided")

        model = load_model(self.model_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['tavg']]  # Only keep the average temperature column
        
        # Split data into train and test
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        temp_input = scaled_test_data[-time_step:]
        forecasted_values = []

        for _ in range(forecast_days):
            temp_input_reshaped = temp_input.reshape(1, time_step, 1)
            pred = model.predict(temp_input_reshaped, verbose=0)
            forecasted_values.append(pred[0, 0])
            temp_input = np.append(temp_input, pred)[-time_step:]

        # Inverse scaling to get actual temperature values
        forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
        forecasted_values_flat = forecasted_values.flatten()

        # Plot forecast
        plt.figure(figsize=(15, 5))
        plt.plot(df.index, df['tavg'], label='Nhiệt độ thực tế', color='red')
        forecast_index = pd.date_range(start=df.index[-1], periods=forecast_days + 1, freq='D')[1:]
        plt.plot(forecast_index, forecasted_values, label=f'Dự báo {forecast_days} ngày tới', color='purple')
        fitted_seri = pd.Series(forecasted_values_flat, index=forecast_index, name='Dự báo nhiệt độ')
        fitted_seri.index.name = "Ngày"
        plt.xlabel('Thời gian')
        plt.ylabel('Nhiệt độ trung bình')
        plt.title(f'Dự báo nhiệt độ trung bình {location_forecasting} trong {forecast_days} ngày tới')
        plt.legend()
        plt.show()

        return fitted_seri,plt
    def check_input(self,input_value):
        if not input_value:  # Nếu giá trị bị bỏ trống
            st.warning(f"Hãy nhập giá trị dự báo keywords ở bên trái trước khi tiếp tục!", icon="⚠️")
            return False
        return True
class DataVisualization:
    def __init__(self, df_selection):
        self.df_selection = df_selection

    def update_fig_layout(self, fig, title):
        """
        Cập nhật bố cục biểu đồ với tiêu đề luôn căn giữa khung hình, bất kể kích thước thay đổi.
        
        :param fig: Đối tượng biểu đồ Plotly
        :param title: Tiêu đề của biểu đồ
        :return: Biểu đồ được cập nhật với bố cục mới
        """
        fig.update_layout(
            title={
                'text': f'<b>{title}</b>',
                'x': 0.5,  # Căn giữa tiêu đề theo chiều ngang
                'xanchor': 'center',  # Cố định tiêu đề vào giữa
                'yanchor': 'top',  # Tiêu đề nằm ở trên cùng
                'y': 0.95,  # Khoảng cách tiêu đề so với khung (cao hơn vùng biểu đồ)
                'font': dict(size=18, color="#333333")  # Font tiêu đề
            },
            autosize=True,  # Tự động điều chỉnh kích thước
            paper_bgcolor='#F5F5F5',  # Màu nền ngoài (xám nhạt)
            plot_bgcolor='#FFFFFF',  # Màu nền trong (trắng)
            margin=dict(l=10, r=10, t=80, b=10),  # Lề đủ rộng cho tiêu đề
            xaxis=dict(
                showline=True,
                linewidth=1.5,
                linecolor="#333333",
                mirror=True,
                gridcolor='#E0E0E0',
                zeroline=False,
                ticks='outside',
                tickfont=dict(size=12, color="#333333")
            ),
            yaxis=dict(
                showline=True,
                linewidth=1.5,
                linecolor="#333333",
                mirror=True,
                gridcolor='#E0E0E0',
                zeroline=False,
                ticks='outside',
                tickfont=dict(size=12, color="#333333")
            )
        )
        return fig
    def plot_job_analysis(self):
        """Plot a bar chart for the distribution of customers by job."""
        job_count = self.df_selection['job'].value_counts().reset_index()
        job_count.columns = ['Công việc', 'Số lượng']

        fig = px.bar(
            job_count,
            x='Số lượng',
            y='Công việc',
            orientation='h',
            color='Số lượng',
            color_continuous_scale=px.colors.sequential.Blues
        )

        fig.update_layout(coloraxis_showscale=False)
        return self.update_fig_layout(fig, 'Số Lượng Khách Hàng Theo Công Việc')
    def plot_gender_pie_chart(self):
        """Plot a pie chart showing customer distribution by gender."""
        gender_count = self.df_selection['sex'].value_counts().reset_index()
        gender_count.columns = ['sex', 'Number of customers']

        fig = px.pie(
            gender_count,
            values='Number of customers',
            names='sex',
            title='<b>Number of customers by Gender</b>',
            color_discrete_sequence=px.colors.sequential.Blues
        )

        return self.update_fig_layout(fig, 'Tỷ Lệ Khách Hàng Theo Giới Tính')
    def plot_location_analysis(self):
        """Plot a bar chart for the number of customers by location."""
        location_count = self.df_selection['address'].value_counts().reset_index()
        location_count.columns = ['Địa điểm', 'Số lượng khách hàng']

        fig = px.bar(
            location_count,
            x='Địa điểm',
            y='Số lượng khách hàng',
            color='Số lượng khách hàng',
            color_continuous_scale=px.colors.sequential.Blues
        )

        fig.update_layout(coloraxis_showscale=False)
        return self.update_fig_layout(fig, 'Số Lượng Khách Hàng Theo Địa Điểm Bán')
    def plot_age_distribution(self):
        """Plot a histogram and normal distribution curve for customer age."""
        ages = self.df_selection['age']  # Dữ liệu tuổi
        hist_data = np.histogram(ages, bins=10)  # Tạo histogram với 10 bins

        fig = go.Figure()

        # Thêm biểu đồ cột
        fig.add_trace(go.Bar(
            x=hist_data[1][:-1],  # Các giá trị biên của bins
            y=hist_data[0],  # Tần số
            marker=dict(
                colorscale='Blues',  # Thang màu xanh
                color=hist_data[0]   # Màu dựa trên tần số
            ),
            name='Number of Customers',
            opacity=0.8
        ))

        # Tính toán phân phối chuẩn
        mu, std = norm.fit(ages)  # Trung bình và độ lệch chuẩn
        x = np.linspace(min(ages), max(ages), 1000)  # 1000 điểm để vẽ đường phân phối
        p = norm.pdf(x, mu, std)  # Mật độ xác suất
        p_scaled = p * max(hist_data[0]) / max(p)  # Chuẩn hóa theo histogram

        # Thêm đường phân phối chuẩn
        fig.add_trace(go.Scatter(
            x=x,
            y=p_scaled,
            mode='lines',
            line=dict(color='Blue', width=2, dash='dash'),
            name='Phân phối chuẩn'
        ))

        # Thêm layout vào đồ thị
        return self.update_fig_layout(fig, 'Phân Bố Khách Hàng Theo Độ Tuổi')
    def plot_revenue_by_product_type(self): 
    # Group and aggregate the revenue by product type
        revenue_by_product_type = self.df_selection.groupby('type')['Doanh thu'].sum().reset_index()
        revenue_by_product_type = revenue_by_product_type.sort_values(by='Doanh thu', ascending=False)

        # Create the pie chart with a blue color theme
        fig_pie = px.pie(
            revenue_by_product_type,
            names='type',
            values='Doanh thu',
            color_discrete_sequence=px.colors.sequential.Blues,  # Updated to use a discrete blue color scale
            title='Tỷ Trọng Doanh Thu Theo Loại Nồng Độ'
        )

        # Update the traces to include percentage and label
        fig_pie.update_traces(textinfo='percent+label')
        # Apply additional layout updates
        return self.update_fig_layout(fig_pie, 'Tỷ Trọng Doanh Thu Theo Loại Nồng Độ')
    def plot_top_products_by_revenue(self, top_n=10):
        df_best_revenue = self.df_selection.groupby('brand')['Doanh thu'].sum().nlargest(top_n).reset_index()
        fig = px.bar(df_best_revenue, 
                     x='brand', 
                     y='Doanh thu',
                     color='Doanh thu',
                     color_continuous_scale=px.colors.sequential.Blues)
        fig.update_layout(coloraxis_showscale=False)
        return self.update_fig_layout(fig, f'Top {top_n} Hãng Bán Chạy Nhất Theo Doanh Thu')
    def plot_monthly_orders_and_products(self):
        self.df_selection['order_time'] = pd.to_datetime(self.df_selection['order_time'])
        self.df_selection['Tháng'] = self.df_selection['order_time'].dt.to_period('M')
        monthly_orders = self.df_selection.groupby('Tháng').agg(
            Số_lượng_đơn_hàng=('order_id', 'nunique'), 
            Tổng_Số_Lượng_Sản_Phẩm=('product_id', 'count')
        ).reset_index()
        monthly_orders['Tháng'] = monthly_orders['Tháng'].dt.to_timestamp()
        long_df = pd.melt(monthly_orders, id_vars=['Tháng'],
                          value_vars=['Số_lượng_đơn_hàng', 'Tổng_Số_Lượng_Sản_Phẩm'],
                          var_name='Category', value_name='Số lượng')
        fig = px.line(long_df,
                      x='Tháng',
                      y='Số lượng',
                      color='Category',
                      labels={'Category': 'Loại Số liệu'},
                      color_discrete_map={'Số_lượng_đơn_hàng': px.colors.sequential.Blues[3], 
                                          'Tổng_Số_Lượng_Sản_Phẩm': px.colors.sequential.Blues[5]},
                      markers=True)
        return self.update_fig_layout(fig, 'Số Đơn Hàng Và Số Lượng Sản Phẩm Theo Tháng')
    def plot_product_by_type(self):
        """Plot the number of products by type."""
        df_product_by_type = self.df_selection.groupby('type').size().reset_index(name='Number of Products')

        fig = px.bar(
            df_product_by_type,
            x='type',
            y='Number of Products',
            color='Number of Products',
            color_continuous_scale=px.colors.sequential.Blues
        )

        return self.update_fig_layout(fig, 'Số Lượng Các Loại Sản Phẩm Theo Nồng Độ')
    def plot_production_location_count(self):
        """Plot the number of products by production location."""
        df_location_count = self.df_selection.groupby('itemLocation').size().reset_index(name='Number of Products')

        fig = px.bar(
            df_location_count,
            x='itemLocation',
            y='Number of Products',
            color='Number of Products',
            color_continuous_scale=px.colors.sequential.Blues
        )

        return self.update_fig_layout(fig, 'Số Lượng Sản Phẩm Theo Địa Điểm Sản Xuất')
    def plot_price_vs_product_type(self):
        """Plot a scatter plot of price vs. product type."""
        fig = px.scatter(
            self.df_selection,
            x='type',
            y='price',
            color='type',
            color_discrete_sequence=px.colors.sequential.Blues
        )

        fig.update_traces(marker=dict(size=10))
        return self.update_fig_layout(fig, 'Mối Tương Quan Giữa Giá Sản Phẩm Và Loại Sản Phẩm')
    def plot_order_value_distribution(self):
        """Plot a box plot for the distribution of order values."""
        fig = px.box(
            self.df_selection,
            y='price',
            color_discrete_sequence=[px.colors.sequential.Blues[3]]
        )

        return self.update_fig_layout(fig, 'Phân Phối Giá Trị Đơn Hàng')
    def plot_orders_by_city(self):
        """Plot a bar chart for the number of orders by city."""
        order_counts = self.df_selection.groupby('address').size().reset_index(name='Number of Orders')

        fig = px.bar(
            order_counts,
            x='address',
            y='Number of Orders',
            color='Number of Orders',
            color_continuous_scale=px.colors.sequential.Blues
        )

        return self.update_fig_layout(fig, 'Số Lượng Đơn Hàng Theo Từng Thành Phố')
    def plot_price_vs_quantity(self):
        """Plot a scatter plot of price vs. quantity sold."""
        color_scale = px.colors.sequential.Blues
        point_color = color_scale[3]

        fig = px.scatter(
            self.df_selection,
            x='price',
            y='quanlity',
            title='<b>Scatter Plot of Product Price vs Quantity Sold</b>',
            labels={'price': 'Product Price', 'quanlity': 'Quantity Sold'},
            size='quanlity',
            color_discrete_sequence=[point_color]
        )

        return self.update_fig_layout(fig, 'Mối Tương Quan Giữa Giá Sản Phẩm Và Số Lượng Bán')
    def calculate_clv(self, customer_id_col, order_id_col, total_col, period='M', retention_rate=0.6, discount_rate=0.1):
        # Tính toán các chỉ số
        average_order_value = self.df_selection.groupby(order_id_col)[total_col].sum().mean()
        purchase_frequency = self.df_selection[order_id_col].nunique() / self.df_selection[customer_id_col].nunique()
        clv = (average_order_value * purchase_frequency * retention_rate) / (1 + discount_rate - retention_rate)

        # Tạo DataFrame kết quả
        clv_data = pd.DataFrame({
            'Chỉ số': ['Giá trị trung bình đơn hàng', 'Tần suất mua hàng', 'Tỷ lệ giữ chân', 'CLV dự đoán'],
            'Giá trị': [average_order_value, purchase_frequency, retention_rate, clv]
        })

        # Tạo biểu đồ
        fig = px.bar(
            clv_data,
            x='Chỉ số',
            y='Giá trị',
            color='Chỉ số',
            text='Giá trị',  # Thêm dữ liệu để hiển thị giá trị
            color_discrete_sequence=px.colors.sequential.Blues
        )

        # Định dạng hiển thị giá trị trên các thanh
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

        # Tắt thanh hiển thị màu
        fig.update_layout(coloraxis_showscale=False)

        return self.update_fig_layout(fig, 'Giá Trị Vòng Đời Khách Hàng (CLV)')
    def plot_customer_return_rate(self, customer_id_col):
        # Calculate the number of returning and new customers
        customer_order_counts = self.df_selection[customer_id_col].value_counts()
        return_customers = customer_order_counts[customer_order_counts > 1].count()
        total_customers = customer_order_counts.count()

        # Define categories and their respective values
        categories = ['Khách hàng quay lại', 'Khách hàng mới']
        values = [(return_customers / total_customers) * 100, 100 - (return_customers / total_customers) * 100]

        # Create a pie chart
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=0.4,
            marker=dict(colors=px.colors.sequential.Blues[2:4])  # Use the first two colors of the Blues color scale
        )])

        # Update the figure layout
        return self.update_fig_layout(fig, 'Tỷ Lệ Khách Hàng Quay Lại')
    def plot_sales_map(self):
        fig = px.scatter_mapbox(
            self.df_selection,
            lat="lat",
            lon="lon",
            hover_name="address",
            hover_data={"Doanh thu": True},
            size="Doanh thu",
            color="Doanh thu",
            color_continuous_scale=px.colors.sequential.Blues,
            size_max=15,
            zoom=5,
           # title="Doanh thu theo thành phố tại Việt Nam"
        )

        # Sử dụng bản đồ Mapbox mặc định
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            title_x=0.5
        )

        return self.update_fig_layout(fig, "Doanh Thu Theo Thành Phố Tại Việt Nam")
def Home(df_selection):
    with st.expander("VIEW ORDER DATASET"):
        showData = st.multiselect(
            'Filter:',
            options=[
                "order_time", "orderdetail_id", "order_id", "customer_id", 
                "product_id", "brand", "type", "price", "quanlity","itemLocation", 
                "weather", "address", "sex", "age", "lat", "lon","Doanh thu"
            ],
            default=[
                "order_time", "orderdetail_id", "order_id", "customer_id", 
                "product_id", "brand", "type", "price", "quanlity","itemLocation", 
                "weather", "address", "sex", "age", "lat", "lon","Doanh thu"
            ]
        )
        
        # Display filtered data
        st.dataframe(df_selection[showData], use_container_width=True)

        # Metrics Calculation
    df_selection['Doanh thu'] = df_selection['price'] * df_selection['quanlity']
    total_revenue = df_selection['Doanh thu'].sum()
    metrics=CustomerMetrics(df_selection)
    clv = metrics.calculate_clv('customer_id', 'order_id', 'Doanh thu')
    aov = metrics.calculate_aov('Doanh thu', 'order_id')
    total_orders = metrics.calculate_total_orders('order_id')
    retention_rate = metrics.calculate_customer_retention_rate('customer_id')
    if not df_selection.empty:  # Check if df_selection is not empty
        df_selection['Doanh thu'] = df_selection['price'] * df_selection['quanlity']
        total_revenue = df_selection['Doanh thu'].sum()
        clv = metrics.calculate_clv('customer_id', 'order_id', 'Doanh thu')
        aov = metrics.calculate_aov('Doanh thu', 'order_id')
        total_orders = metrics.calculate_total_orders('order_id')
        retention_rate = metrics.calculate_customer_retention_rate('customer_id')
    else:
        total_revenue = 0
        clv = 0
        aov = 0
        total_orders = 0
        retention_rate = 0
        # Display metrics
    st.header('Metrics Overview')
    total1, total2, total3, total4, total5 = st.columns(5, gap='small')
        
    with total1:
        st.info("Tổng doanh thu",icon="💰")
        st.metric(label="",value=f"{total_revenue:.2f} VND")
    with total2:
        st.info("Giá trị CLV",icon="💰")
        st.metric(label="",value=f"{clv:.2f} VND")
    with total3:
        st.info("Giá trị AOV",icon="💰")
        st.metric(label="",value=f"{aov:.2f} VND")
    with total4:
        st.info("Tổng số đơn hàng",icon="🛒")
        st.metric(label="",value=f"{total_orders:.0f}")
    with total5:
        st.info('Tỷ lệ giữ chân khách hàng',icon="🏃🏻")
        st.metric(label="", value=f"{retention_rate:.2f}%")

def selection(brand,type_perfume,selected_price_range,item_location,weather,address,sex,selected_age_range,month_review_filter,year_review_filter):
    df_selection = filtered_data.query(
        "brand in @brand & "
        "type in @type_perfume & "
        "price >= @selected_price_range[0] & price <= @selected_price_range[1] & "
        "itemLocation in @item_location & "
        "weather in @weather & "
        "address in @address & "
        "sex in @sex & "
        "age >= @selected_age_range[0] & age <= @selected_age_range[1] & "
        "order_time.dt.month in @month_review_filter & "
        "order_time.dt.year in @year_review_filter"
    )
    return df_selection
def filter_main():
    filtermanager=FilterManager(filtered_data)
    brand_ft,type_perfume_ft,selected_price_range_ft,item_location_ft,weather_ft,address_ft,sex_ft,selected_age_range_ft,month_review_filter_ft,year_review_filter_ft=filtermanager.render_filters()
    df_selection=selection(brand_ft,type_perfume_ft,selected_price_range_ft,item_location_ft,weather_ft,address_ft,sex_ft,selected_age_range_ft,month_review_filter_ft,year_review_filter_ft)
    
server = r'DESKTOP-Q6B5CSD\NAMNH'
database = 'Capstone2'
username = 'sa'
password = 'Nam@15092003'
db_manager = DatabaseManager(server, database, username, password)
# Kết nối đến database
db_manager.connect()
query = '''
SELECT o.order_time, od.orderdetail_id, o.order_id, c.customer_id, c.job, 
       p.product_id, p.brand, p.type, p.price, od.quanlity, p.itemLocation,
       p.weather, c.address, c.sex, c.age
FROM data_khachhang c
INNER JOIN Orders o ON c.customer_id = o.customer_id
INNER JOIN Order_detail od ON o.order_id = od.order_id
INNER JOIN data_sanpham p ON od.sanpham_id = p.product_id'''
query1 = '''SELECT * from data_toado'''
query2 = '''SELECT * from data_khachhang'''

# Fetch data from SQL Server
with db_manager.engine.connect() as connection:
    df = pd.read_sql(query, connection)
    toado= pd.read_sql(query1, connection)
    diadiem=pd.read_sql(query2, connection)
# Ensure order_time is in datetime format
df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')

# Handle address data
diadiem['address']=diadiem['address'].str.split(',').str[-1].str.strip()
df['address'] = df['address'].str.split(',').str[-1].str.strip()
filtered_data = pd.merge(df, toado, left_on='address', right_on='city', how='inner').drop(columns=['city'])

# Calculate revenue
filtered_data['Doanh thu'] = filtered_data['price'] * filtered_data['quanlity']
def sideBar():
    filtermanager=FilterManager(filtered_data)
    if selected=="Dashboard" or selected == "Analytics":
        brand_ft,type_perfume_ft,selected_price_range_ft,item_location_ft,weather_ft,address_ft,sex_ft,selected_age_range_ft,month_review_filter_ft,year_review_filter_ft=filtermanager.render_filters()
        df_selection=selection(brand_ft,type_perfume_ft,selected_price_range_ft,item_location_ft,weather_ft,address_ft,sex_ft,selected_age_range_ft,month_review_filter_ft,year_review_filter_ft)
        analytics = DataVisualization(df_selection)
        if selected=="Dashboard":
            Home(df_selection)
            st.plotly_chart(analytics.plot_job_analysis(), use_container_width=True)
            st.plotly_chart(analytics.plot_location_analysis(), use_container_width=True)
            st.plotly_chart(analytics.plot_revenue_by_product_type(), use_container_width=True)
            st.plotly_chart(analytics.plot_top_products_by_revenue(), use_container_width=True)
            st.plotly_chart(analytics.plot_monthly_orders_and_products(), use_container_width=True)
            st.plotly_chart(analytics.plot_order_value_distribution(), use_container_width=True)
            st.plotly_chart(analytics.calculate_clv( 'customer_id', 'order_id', 'price'), use_container_width=True)
            st.plotly_chart(analytics.plot_customer_return_rate( 'customer_id'), use_container_width=True)
            st.plotly_chart(analytics.plot_sales_map(), use_container_width=True)
        if selected == "Analytics":
            with st.expander("Tổng Quan Về Khách Hàng"):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(analytics.plot_job_analysis(), use_container_width=True)
                with col2:
                    st.plotly_chart(analytics.plot_gender_pie_chart(), use_container_width=True)

                # Hàng thứ hai với 2 biểu đồ khác
                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(analytics.plot_location_analysis(), use_container_width=True)
                with col4:
                    st.plotly_chart(analytics.plot_age_distribution(), use_container_width=True)
            with st.expander("Tổng Quan Về Sản Phẩm"):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(analytics.plot_top_products_by_revenue(), use_container_width=True)
                with col2:
                    st.plotly_chart(analytics.plot_product_by_type(), use_container_width=True)

                # Hàng thứ hai với 2 biểu đồ
                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(analytics.plot_production_location_count(), use_container_width=True)
                with col4:
                    st.plotly_chart(analytics.plot_price_vs_product_type(), use_container_width=True)
            with st.container():
                st.subheader("Tổng Quan Về Đơn Hàng")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(analytics.plot_order_value_distribution(), use_container_width=True)
                with col2:
                    st.plotly_chart(analytics.plot_orders_by_city(), use_container_width=True)

                # Hàng thứ hai với 2 biểu đồ
                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(analytics.plot_price_vs_quantity(), use_container_width=True)
                with col4:
                    st.plotly_chart(analytics.plot_sales_map(), use_container_width=True)
    if selected == "Forecasting":
        st.subheader("Trang dự báo và đề xuất")
        # Lấy thông tin bộ lọc
        key_forecasting, location_forecasting, time_keyword_forecasting, time_weather_forecasting = filtermanager.filter_forecasting()
        url = "https://semrush-keyword-magic-tool.p.rapidapi.com/keyword-research"
        api_key = "510309684fmsha6871e657668cfep13e5a1jsn20f6f5b9a769"
        api_host = "semrush-keyword-magic-tool.p.rapidapi.com"
        model_path = f'{location_forecasting}_best_weather_model.keras'
        query_location = f'''SELECT * FROM [dbo].[data_[{location_forecasting}]]]'''
        # Khởi tạo lớp Forecasting
        forecasting = Forecasting(url, api_key, api_host, model_path)
        forecast_days = int(time_weather_forecasting)
        result_df=None
        predictions=None
        if key_forecasting:
            try:
                result_df = forecasting.get_API(key_forecasting)
                if result_df is None or result_df.empty:
                    st.write("API không trả về dữ liệu hợp lệ.")
            except Exception as e:
                st.write(f"Lỗi khi gọi API: {e}")
        else:
            st.write("Key forecasting chưa được cung cấp. Vui lòng nhập để tiếp tục.")
        try:
            n_pred_periods = int(time_keyword_forecasting)
        except ValueError:
            forecasting.check_input(time_keyword_forecasting)
            st.write("Thời gian dự báo (time_keyword_forecasting) không hợp lệ. Vui lòng nhập số nguyên.")
            return
        col1, col2 = st.columns([3.5, 1.5])
        with col1:
            if result_df is not None:
                predictions, plot_obj_keyword = forecasting.forecast_with_sarima(result_df, n_pred_periods,key_forecasting)
                if not predictions.empty:
                    st.pyplot(plot_obj_keyword)  # Hiển thị biểu đồ dự đoán
                else:
                    st.write(f"{key_forecasting} không có sự tìm kiếm")
            with db_manager.engine.connect() as connection:
                df = pd.read_sql(query_location, connection)
            forecastlist, plt_obj_weather = forecasting.forecast_weather(df, forecast_days,location_forecasting, time_step=30)
            st.pyplot(plt_obj_weather)
        with col2:
            try:
                if not predictions.empty:
                    st.write("Kết quả dự đoán Keyword:")  # Hiển thị kết quả dự đoán trong cột chiếm 20%
                    st.write(predictions)
            except:
                None
            st.write(f"Kết quả dự đoán thời tiết trong {time_weather_forecasting} ngày ở {location_forecasting}")
            st.write(forecastlist)

sideBar()
hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
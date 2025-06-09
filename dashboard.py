# dashboard_integrated.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

# Cấu hình trang
st.set_page_config(
    page_title="Dashboard Quản lý Tổ xe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef, #f8f9fa);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-text {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
        text-align: center;
        flex: 1;
    }
    .logo-container {
        flex-shrink: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric > label {
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

def calculate_distance_for_vehicle(vehicle_code, vehicle_data):
    """Tính quãng đường cho một xe cụ thể"""
    
    # Sắp xếp theo ngày ghi nhận, sau đó thời gian bắt đầu (nếu có)
    if 'Thời gian bắt đầu' in vehicle_data.columns and vehicle_data['Thời gian bắt đầu'].notna().any():
        vehicle_data = vehicle_data.sort_values(['Ngày ghi nhận', 'Thời gian bắt đầu']).reset_index(drop=True)
    else:
        vehicle_data = vehicle_data.sort_values(['Ngày ghi nhận']).reset_index(drop=True)
    
    # Khởi tạo cột quãng đường với kiểu dữ liệu float64
    vehicle_data['Quãng đường (km)'] = np.float64(np.nan)
    
    # Tính quãng đường với kiểm tra hợp lệ
    for i in range(1, len(vehicle_data)):  # Bắt đầu từ chuyến thứ 2
        current_odo = vehicle_data.iloc[i]['Chỉ số đồng hồ']
        prev_odo = vehicle_data.iloc[i-1]['Chỉ số đồng hồ']
        
        distance = current_odo - prev_odo
        
        # Kiểm tra điều kiện hợp lệ:
        # 1. Quãng đường >= 0 (không giảm)
        # 2. Quãng đường <= 2000km (không quá lớn - có thể là lỗi dữ liệu)
        if distance >= 0 and distance <= 640:
            vehicle_data.iloc[i, vehicle_data.columns.get_loc('Quãng đường (km)')] = np.float64(distance)
        # Nếu quãng đường > 2000km hoặc < 0, bỏ qua (để NaN)
    
    return vehicle_data

@st.cache_data(ttl=60)  # Cache 1 phút
def load_data():
    """Đọc dữ liệu từ file Excel local và tính toán quãng đường"""
    
    # Đường dẫn file Excel
    file_path = 'Dashboard DHYD_ver2-4.xlsx'
    
    try:
        # Đọc sheet "Dữ liệu gộp"
        df = pd.read_excel(file_path, sheet_name="Dữ liệu gộp")
        
        # Loại bỏ các cột không cần thiết
        columns_to_drop = ["Sheet gốc", "Dòng gốc", "Timestamp"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Đổi tên cột để dễ xử lý
        expected_columns = [
            "STT", "Mã xe", "Email", "Thời gian bắt đầu", "Thời gian kết thúc",
            "Điểm đón", "Điểm đến", "Phân loại công tác", "Chỉ số đồng hồ",
            "Chi tiết chuyến xe", "Doanh thu", "Ghi chú", "Ngày ghi nhận", "Thời gian chạy (phút)"
        ]
        
        # Chỉ đổi tên các cột có sẵn
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns + list(df.columns[len(expected_columns):])
        else:
            df.columns = expected_columns[:len(df.columns)]
        
        # Xử lý dữ liệu ngày tháng
        if "Ngày ghi nhận" in df.columns:
            def parse_date(date_val):
                if pd.isna(date_val):
                    return pd.NaT
                
                if isinstance(date_val, (pd.Timestamp, datetime)):
                    return pd.to_datetime(date_val)
                
                if isinstance(date_val, str):
                    date_val = date_val.strip()
                    
                    for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%y', '%d/%m/%y']:
                        try:
                            return pd.to_datetime(date_val, format=fmt)
                        except:
                            continue
                    
                    try:
                        return pd.to_datetime(date_val, dayfirst=True)
                    except:
                        return pd.NaT
                
                try:
                    return pd.to_datetime(date_val)
                except:
                    return pd.NaT
            
            df["Ngày ghi nhận"] = df["Ngày ghi nhận"].apply(parse_date)
        
        # Xử lý thời gian bắt đầu
        if "Thời gian bắt đầu" in df.columns:
            df["Thời gian bắt đầu"] = pd.to_datetime(df["Thời gian bắt đầu"], errors='coerce')
        
        # Xử lý chỉ số đồng hồ
        if "Chỉ số đồng hồ" in df.columns:
            def clean_odo(val):
                if pd.isna(val):
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    import re
                    val_clean = re.sub(r'[^\d.]', '', str(val))
                    try:
                        return float(val_clean) if val_clean else np.nan
                    except:
                        return np.nan
                try:
                    return float(val)
                except:
                    return np.nan
            
            df["Chỉ số đồng hồ"] = df["Chỉ số đồng hồ"].apply(clean_odo)
        
        # Xử lý doanh thu
        if "Doanh thu" in df.columns:
            def clean_revenue(val):
                if pd.isna(val):
                    return 0
                if isinstance(val, str):
                    val = val.replace(',', '').replace(' ', '').replace('.', '')
                    import re
                    numbers = re.findall(r'\d+', str(val))
                    if numbers:
                        return float(''.join(numbers))
                return float(val) if str(val).replace('.', '').isdigit() else 0
            
            df["Doanh thu"] = df["Doanh thu"].apply(clean_revenue)
        
        # Xử lý thời gian chạy
        if "Thời gian chạy (phút)" in df.columns:
            def clean_time(val):
                if pd.isna(val):
                    return 0
                if isinstance(val, str):
                    import re
                    numbers = re.findall(r'\d+', str(val))
                    if numbers:
                        return float(numbers[0])
                return float(val) if str(val).replace('.', '').isdigit() else 0
            
            df["Thời gian chạy (phút)"] = df["Thời gian chạy (phút)"].apply(clean_time)
            df["Thời gian chạy (giờ)"] = df["Thời gian chạy (phút)"] / 60
        
        # Tạo cột tháng năm và ngày
        if "Ngày ghi nhận" in df.columns and not df["Ngày ghi nhận"].isna().all():
            df["Tháng"] = df["Ngày ghi nhận"].dt.to_period('M').astype(str)
            df["Ngày"] = df["Ngày ghi nhận"].dt.date
        
        # TÍCH HỢP TÍNH TOÁN QUÃNG ĐƯỜNG
        # Lọc dữ liệu hợp lệ cho tính toán quãng đường
        df_valid_for_distance = df[
            df['Mã xe'].notna() & 
            df['Ngày ghi nhận'].notna() & 
            df['Chỉ số đồng hồ'].notna()
        ].copy()
        
        # Tính toán quãng đường cho từng xe
        if len(df_valid_for_distance) > 0:
            results = []
            for vehicle in sorted(df_valid_for_distance['Mã xe'].unique()):
                vehicle_data = df_valid_for_distance[df_valid_for_distance['Mã xe'] == vehicle].copy()
                result = calculate_distance_for_vehicle(vehicle, vehicle_data)
                results.append(result)
            
            # Gộp kết quả tính toán quãng đường
            df_with_distance = pd.concat(results, ignore_index=True)
            
            # Đảm bảo cột quãng đường có kiểu dữ liệu đúng
            df_with_distance['Quãng đường (km)'] = pd.to_numeric(df_with_distance['Quãng đường (km)'], errors='coerce')
            
            # Merge quãng đường vào dataframe chính
            merge_cols = ['STT', 'Mã xe', 'Ngày ghi nhận', 'Chỉ số đồng hồ']
            available_merge_cols = [col for col in merge_cols if col in df.columns and col in df_with_distance.columns]
            
            if available_merge_cols:
                df = df.merge(
                    df_with_distance[available_merge_cols + ['Quãng đường (km)']],
                    on=available_merge_cols,
                    how='left'
                )
                
                # Đảm bảo cột trong df chính cũng có kiểu dữ liệu đúng
                df['Quãng đường (km)'] = pd.to_numeric(df['Quãng đường (km)'], errors='coerce')
        
        return df
        
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy file: {file_path}")
        st.info("""
        💡 **Kiểm tra:**
        1. File có tồn tại tại đường dẫn trên không?
        2. Tên file có chính xác không? (Dashboard DHYD_ver2-4.xlsx)
        3. File có nằm trong thư mục Downloads không?
        """)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc dữ liệu: {e}")
        return pd.DataFrame()

def main():
    # Header với logo và tiêu đề trên cùng một hàng
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Tạo layout flex cho logo và text
    col_logo, col_text = st.columns([1, 6])
    
    with col_logo:
        # Kiểm tra logo
        logo_paths = [
            "assets/logo.png",
            "logo.png",
            "images/logo.png"
        ]
        
        logo_found = False
        for path in logo_paths:
            if os.path.exists(path):
                try:
                    st.image(path, width=80)
                    logo_found = True
                    break
                except:
                    continue
        
        if not logo_found:
            st.markdown('<div style="font-size: 3rem; text-align: center;">🏥</div>', unsafe_allow_html=True)
    
    with col_text:
        st.markdown('<h1 class="header-text">Dashboard Quản lý Phương tiện vận chuyển tại Bệnh viện Đại học Y Dược TP.HCM </h1>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Đọc dữ liệu từ file Excel local
    with st.spinner("📁 Đang tải dữ liệu từ file Excel và tính toán quãng đường..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Không thể đọc dữ liệu.")
        st.info("""
        💡 **Hướng dẫn khắc phục:**
        1. Đảm bảo file `Dashboard DHYD_ver2-4.xlsx` nằm trong thư mục Downloads
        2. Kiểm tra file có sheet tên "Dữ liệu gộp"
        3. Đảm bảo file không bị hỏng
        
        📁 **Đường dẫn file hiện tại:**
        `/Users/quang/Downloads/Dashboard DHYD_ver2-4.xlsx`
        """)
        return
    
    # Khởi tạo các biến toàn cục cho hàm
    date_range = None
    df_filtered_by_time = df.copy()
    
    # Hiển thị thông tin file và debug
    st.sidebar.header("📄 Thông tin dữ liệu")
    st.sidebar.success(f"✅ Đã tải {len(df)} bản ghi")
    st.sidebar.info(f"📁 File: Dashboard DHYD_ver2-4.xlsx")
    st.sidebar.info(f"🔄 Cập nhật: {datetime.now().strftime('%H:%M:%S')}")
    
    # Kiểm tra tính năng quãng đường
    if "Quãng đường (km)" in df.columns:
        distance_records = df["Quãng đường (km)"].notna().sum()
        total_distance = df["Quãng đường (km)"].sum()
        st.sidebar.success(f"🛣️ Đã tính {distance_records} quãng đường")
        st.sidebar.info(f"📏 Tổng: {total_distance:.1f} km")
    else:
        st.sidebar.warning("⚠️ Chưa tính được quãng đường")
    
    # Debug: Hiển thị các cột có sẵn
    st.sidebar.write("🔍 **Các cột trong dữ liệu:**")
    for col in df.columns:
        st.sidebar.write(f"- {col}")
    
    # Kiểm tra và tạo cột "Tên tài xế" nếu chưa có
    if "Tên tài xế" not in df.columns and "Email" in df.columns:
        driver_names = {
            "ngochai191974@gmail.com": "Ngọc Hải",
            "phongthai230177@gmail.com": "Thái Phong", 
            "dunglamlong@gmail.com": "Long Dũng",
            "trananhtuan461970@gmail.com": "Anh Tuấn",
            "thanhdungvo29@gmail.com": "Thanh Dũng",
            "duck79884@gmail.com": "Đức",
            "ngohoangxuyen@gmail.com": "Hoàng Xuyên",
            "hodinhxuyen@gmail.com": "Đình Xuyên",
            "nvhung1981970@gmail.com": "Văn Hùng",
            "thanggptk21@gmail.com": "Văn Thảo",
            "nguyenhung091281@gmail.com": "Nguyễn Hùng",
            "nguyemthanhtrung12345@gmail.com": "Thành Trung",
            "nguyenhungumc@gmail.com": "Nguyễn Hùng",
            "dvo567947@gmail.com": "Tài xế khác",
            "traannhtuan461970@gmail.com": "Anh Tuấn",
            "hoanganhsie1983@gmail.com": "Hoàng Anh",
            "hoanganhsieumc@gmail.com": "Hoàng Anh",
            "thaonguyenvan860@gmail.com": "Văn Thảo"
        }
        df["Tên tài xế"] = df["Email"].map(driver_names).fillna(df["Email"])
        st.sidebar.success("✅ Đã tạo cột 'Tên tài xế'")
    
    # Hiển thị thông tin ngày tháng
    if "Ngày ghi nhận" in df.columns and not df["Ngày ghi nhận"].isna().all():
        st.sidebar.success(f"📅 Dữ liệu từ: {df['Ngày ghi nhận'].min().strftime('%d/%m/%Y')} đến {df['Ngày ghi nhận'].max().strftime('%d/%m/%Y')}")
    
    # Nút làm mới dữ liệu
    if st.sidebar.button("🔄 Làm mới dữ liệu"):
        st.cache_data.clear()
        st.rerun()
    
    # Sidebar - Bộ lọc
    st.sidebar.header("🔍 Bộ lọc dữ liệu")
    
    # Lọc theo thời gian
    if "Ngày ghi nhận" in df.columns and not df["Ngày ghi nhận"].isna().all():
        st.sidebar.subheader("📅 Lọc theo thời gian")
        
        date_filter_type = st.sidebar.radio(
            "Chọn cách lọc:",
            ["Toàn bộ dữ liệu", "Khoảng thời gian", "Tháng/Năm cụ thể"],
            index=0
        )
        
        if date_filter_type == "Khoảng thời gian":
            date_range = st.sidebar.date_input(
                "Chọn khoảng thời gian:",
                value=[df["Ngày ghi nhận"].min().date(), df["Ngày ghi nhận"].max().date()],
                min_value=df["Ngày ghi nhận"].min().date(),
                max_value=df["Ngày ghi nhận"].max().date(),
                help="Chọn ngày bắt đầu và ngày kết thúc"
            )
            
            if len(date_range) == 2:
                df_filtered_by_time = df[
                    (df["Ngày ghi nhận"].dt.date >= date_range[0]) &
                    (df["Ngày ghi nhận"].dt.date <= date_range[1])
                ]
        
        elif date_filter_type == "Tháng/Năm cụ thể":
            try:
                available_months = df["Ngày ghi nhận"].dt.to_period('M').dropna().unique()
                available_months = sorted([str(month) for month in available_months])
                
                selected_months = st.sidebar.multiselect(
                    "Chọn tháng/năm:",
                    options=available_months,
                    default=available_months,
                    help="Có thể chọn nhiều tháng"
                )
                
                if selected_months:
                    selected_periods = [pd.Period(month) for month in selected_months]
                    min_date = min(selected_periods).start_time.date()
                    max_date = max(selected_periods).end_time.date()
                    date_range = [min_date, max_date]
                    
                    df_filtered_by_time = df[
                        (df["Ngày ghi nhận"].dt.date >= min_date) &
                        (df["Ngày ghi nhận"].dt.date <= max_date)
                    ]
            except Exception as e:
                st.sidebar.error(f"Lỗi xử lý tháng/năm: {e}")
                df_filtered_by_time = df.copy()
        
        # Hiển thị thống kê về dữ liệu đã lọc
        if date_filter_type != "Toàn bộ dữ liệu" and date_range and len(date_range) == 2:
            st.sidebar.info(f"""
            📊 **Dữ liệu trong khoảng thời gian:**
            - Từ: {date_range[0].strftime('%d/%m/%Y')}
            - Đến: {date_range[1].strftime('%d/%m/%Y')}
            - Số bản ghi: {len(df_filtered_by_time):,}/{len(df):,}
            - Tỷ lệ: {(len(df_filtered_by_time)/len(df)*100):.1f}%
            """)
            st.sidebar.success(f"📅 Đang phân tích: {date_range[0].strftime('%d/%m/%Y')} - {date_range[1].strftime('%d/%m/%Y')}")
        else:
            st.sidebar.info(f"📊 **Toàn bộ dữ liệu:** {len(df):,} bản ghi")
    
    # Các bộ lọc khác
    vehicles = st.sidebar.multiselect(
        "Chọn xe:",
        options=sorted(df_filtered_by_time["Mã xe"].dropna().unique()),
        default=sorted(df_filtered_by_time["Mã xe"].dropna().unique()),
        help="Chọn xe muốn phân tích"
    )
    
    if "Tên tài xế" in df_filtered_by_time.columns:
        drivers = st.sidebar.multiselect(
            "Chọn tài xế:",
            options=sorted(df_filtered_by_time["Tên tài xế"].dropna().unique()),
            default=sorted(df_filtered_by_time["Tên tài xế"].dropna().unique()),
            help="Chọn tài xế muốn phân tích"
        )
    else:
        drivers = sorted(df_filtered_by_time["Email"].dropna().unique()) if "Email" in df_filtered_by_time.columns else []
        st.sidebar.warning("⚠️ Sử dụng Email thay vì tên tài xế")
    
    work_types = st.sidebar.multiselect(
        "Loại công tác:",
        options=sorted(df_filtered_by_time["Phân loại công tác"].dropna().unique()),
        default=sorted(df_filtered_by_time["Phân loại công tác"].dropna().unique()),
        help="Chọn loại công tác muốn phân tích"
    )
    
    # Áp dụng tất cả các bộ lọc
    if "Tên tài xế" in df_filtered_by_time.columns:
        filtered_df = df_filtered_by_time[
            (df_filtered_by_time["Mã xe"].isin(vehicles)) &
            (df_filtered_by_time["Tên tài xế"].isin(drivers)) &
            (df_filtered_by_time["Phân loại công tác"].isin(work_types) | df_filtered_by_time["Phân loại công tác"].isna())
        ]
    else:
        filtered_df = df_filtered_by_time[
            (df_filtered_by_time["Mã xe"].isin(vehicles)) &
            (df_filtered_by_time["Phân loại công tác"].isin(work_types) | df_filtered_by_time["Phân loại công tác"].isna())
        ]
    
    # Metrics tổng quan (TÍCH HỢP QUÃNG ĐƯỜNG)
    st.header("📊 Tổng quan hoạt động")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            label="Tổng số chuyến",
            value=f"{len(filtered_df):,}",
            delta=f"/{len(df):,} chuyến"
        )
    
    with col2:
        total_revenue = filtered_df["Doanh thu"].sum()
        st.metric(
            label="Tổng doanh thu",
            value=f"{total_revenue:,.0f} VNĐ"
        )
    
    with col3:
        total_hours = filtered_df["Thời gian chạy (giờ)"].sum()
        st.metric(
            label="Tổng giờ chạy",
            value=f"{total_hours:.1f} giờ"
        )
    
    with col4:
        # METRIC MỚI: QUÃNG ĐƯỜNG
        if "Quãng đường (km)" in filtered_df.columns:
            total_distance = filtered_df["Quãng đường (km)"].sum()
            st.metric(
                label="Tổng quãng đường",
                value=f"{total_distance:.1f} km",
                help="Tổng quãng đường đã di chuyển"
            )
        else:
            st.metric(
                label="Tổng quãng đường",
                value="N/A",
                help="Chưa tính được quãng đường"
            )
    
    with col5:
        avg_revenue_per_trip = total_revenue / len(filtered_df) if len(filtered_df) > 0 else 0
        st.metric(
            label="Doanh thu TB/chuyến",
            value=f"{avg_revenue_per_trip:,.0f} VNĐ"
        )
    
    with col6:
        num_vehicles = filtered_df["Mã xe"].nunique()
        st.metric(
            label="Số xe hoạt động",
            value=f"{num_vehicles}",
            delta=f"/{df['Mã xe'].nunique()} xe"
        )
    
    # THÊM SECTION PHÂN TÍCH QUÃNG ĐƯỜNG
    if "Quãng đường (km)" in filtered_df.columns:
        st.header("🛣️ Phân tích quãng đường di chuyển")
        
        # Metrics quãng đường chi tiết
        col1, col2, col3, col4 = st.columns(4)
        
        distance_data = filtered_df[filtered_df["Quãng đường (km)"].notna()]
        
        with col1:
            avg_distance = distance_data["Quãng đường (km)"].mean() if not distance_data.empty else 0
            st.metric(
                label="Quãng đường TB/chuyến",
                value=f"{avg_distance:.1f} km"
            )
        
        with col2:
            max_distance = distance_data["Quãng đường (km)"].max() if not distance_data.empty else 0
            st.metric(
                label="Chuyến xa nhất",
                value=f"{max_distance:.1f} km"
            )
        
        with col3:
            trips_with_distance = len(distance_data)
            total_trips = len(filtered_df)
            coverage = (trips_with_distance / total_trips * 100) if total_trips > 0 else 0
            st.metric(
                label="Tỷ lệ tính được QĐ",
                value=f"{coverage:.1f}%",
                delta=f"{trips_with_distance}/{total_trips} chuyến"
            )
        
        with col4:
            if not distance_data.empty and total_hours > 0:
                avg_speed = distance_data["Quãng đường (km)"].sum() / total_hours
                st.metric(
                    label="Tốc độ TB",
                    value=f"{avg_speed:.1f} km/h",
                    help="Tốc độ trung bình = Tổng km / Tổng giờ"
                )
            else:
                st.metric(
                    label="Tốc độ TB",
                    value="N/A"
                )
        
        # Biểu đồ quãng đường
        tab1, tab2, tab3 = st.tabs(["📊 Theo xe", "📅 Theo thời gian", "📋 Chi tiết"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Quãng đường theo xe
                distance_by_vehicle = distance_data.groupby("Mã xe")["Quãng đường (km)"].sum().sort_values(ascending=False)
                if not distance_by_vehicle.empty:
                    fig_distance = px.bar(
                        x=distance_by_vehicle.index,
                        y=distance_by_vehicle.values,
                        title="Tổng quãng đường theo xe",
                        labels={"x": "Mã xe", "y": "Quãng đường (km)"},
                        color=distance_by_vehicle.values,
                        color_continuous_scale="Blues"
                    )
                    fig_distance.update_layout(height=400)
                    st.plotly_chart(fig_distance, use_container_width=True)
            
            with col2:
                # Quãng đường trung bình theo xe
                avg_distance_by_vehicle = distance_data.groupby("Mã xe")["Quãng đường (km)"].mean().sort_values(ascending=False)
                if not avg_distance_by_vehicle.empty:
                    fig_avg_distance = px.bar(
                        x=avg_distance_by_vehicle.index,
                        y=avg_distance_by_vehicle.values,
                        title="Quãng đường trung bình/chuyến theo xe",
                        labels={"x": "Mã xe", "y": "Quãng đường TB (km)"},
                        color=avg_distance_by_vehicle.values,
                        color_continuous_scale="Greens"
                    )
                    fig_avg_distance.update_layout(height=400)
                    st.plotly_chart(fig_avg_distance, use_container_width=True)
        
        with tab2:
            if "Tháng" in distance_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Quãng đường theo tháng
                    monthly_distance = distance_data.groupby("Tháng")["Quãng đường (km)"].sum()
                    if not monthly_distance.empty:
                        fig_monthly_distance = px.line(
                            x=monthly_distance.index,
                            y=monthly_distance.values,
                            title="Xu hướng quãng đường theo tháng",
                            labels={"x": "Tháng", "y": "Quãng đường (km)"},
                            markers=True
                        )
                        fig_monthly_distance.update_layout(height=400)
                        st.plotly_chart(fig_monthly_distance, use_container_width=True)
                
                with col2:
                    # Số chuyến có quãng đường theo tháng
                    monthly_trips_with_distance = distance_data.groupby("Tháng").size()
                    if not monthly_trips_with_distance.empty:
                        fig_monthly_trips = px.bar(
                            x=monthly_trips_with_distance.index,
                            y=monthly_trips_with_distance.values,
                            title="Số chuyến có quãng đường theo tháng",
                            labels={"x": "Tháng", "y": "Số chuyến"},
                            color=monthly_trips_with_distance.values,
                            color_continuous_scale="Oranges"
                        )
                        fig_monthly_trips.update_layout(height=400)
                        st.plotly_chart(fig_monthly_trips, use_container_width=True)
        
        with tab3:
            # Bảng chi tiết quãng đường
            st.subheader("📋 Thống kê chi tiết quãng đường theo xe")
            
            if not distance_data.empty:
                distance_summary = distance_data.groupby("Mã xe").agg({
                    "Quãng đường (km)": ["count", "sum", "mean", "max"],
                    "Doanh thu": "sum",
                    "Thời gian chạy (giờ)": "sum"
                }).round(2)
                
                distance_summary.columns = [
                    "Số chuyến có QĐ", "Tổng QĐ (km)", "QĐ TB (km)", "QĐ Max (km)", 
                    "Tổng doanh thu", "Tổng giờ chạy"
                ]
                
                # Tính hiệu quả
                distance_summary["Tốc độ TB (km/h)"] = (
                    distance_summary["Tổng QĐ (km)"] / distance_summary["Tổng giờ chạy"]
                ).round(1)
                
                distance_summary["Doanh thu/km"] = (
                    distance_summary["Tổng doanh thu"] / distance_summary["Tổng QĐ (km)"]
                ).round(0)
                
                # Sắp xếp theo tổng quãng đường
                distance_summary = distance_summary.sort_values("Tổng QĐ (km)", ascending=False)
                
                st.data_editor(
                    distance_summary.style.format({
                        "Tổng QĐ (km)": "{:.1f}",
                        "QĐ TB (km)": "{:.1f}",
                        "QĐ Max (km)": "{:.1f}",
                        "Tổng doanh thu": "{:,.0f}",
                        "Tổng giờ chạy": "{:.1f}",
                        "Tốc độ TB (km/h)": "{:.1f}",
                        "Doanh thu/km": "{:,.0f}"
                    }),
                    use_container_width=True,
                    disabled=True,
                    hide_index=False
                )
                
                # Phân tích bất thường
                st.subheader("🔍 Phân tích dữ liệu bất thường")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Chuyến quá dài (giảm ngưỡng xuống 300km thay vì 2000km để phát hiện sớm)
                    long_trips = distance_data[distance_data["Quãng đường (km)"] > 300]
                    if not long_trips.empty:
                        st.warning(f"⚠️ **{len(long_trips)} chuyến có quãng đường > 300km:**")
                        display_long = long_trips[["Mã xe", "Ngày ghi nhận", "Quãng đường (km)", "Điểm đến"]].head(5).copy()
                        # Format cột quãng đường cho hiển thị
                        display_long["Quãng đường (km)"] = display_long["Quãng đường (km)"].apply(lambda x: f"{x:.1f} km")
                        st.data_editor(display_long, use_container_width=True, disabled=True, hide_index=True)
                        
                        # Cảnh báo nếu có chuyến gần ngưỡng 2000km
                        very_long_trips = distance_data[distance_data["Quãng đường (km)"] > 1000]
                        if not very_long_trips.empty:
                            st.error(f"🚨 **{len(very_long_trips)} chuyến có quãng đường > 1000km - cần kiểm tra dữ liệu!**")
                    else:
                        st.success("✅ Không có chuyến nào quá dài (>300km)")
                
                with col2:
                    # Chuyến quá ngắn
                    short_trips = distance_data[distance_data["Quãng đường (km)"] < 1]
                    if not short_trips.empty:
                        st.warning(f"⚠️ **{len(short_trips)} chuyến có quãng đường < 1km:**")
                        display_short = short_trips[["Mã xe", "Ngày ghi nhận", "Quãng đường (km)", "Điểm đến"]].head(5).copy()
                        # Format cột quãng đường cho hiển thị
                        display_short["Quãng đường (km)"] = display_short["Quãng đường (km)"].apply(lambda x: f"{x:.1f} km")
                        st.data_editor(display_short, use_container_width=True, disabled=True, hide_index=True)
                    else:
                        st.success("✅ Không có chuyến nào quá ngắn (<1km)")
                
                # Thông báo về việc lọc dữ liệu > 2000km
                total_trips_with_odo = len(df_valid_for_distance) if 'df_valid_for_distance' in locals() else 0
                trips_with_distance = len(distance_data)
                if total_trips_with_odo > trips_with_distance:
                    filtered_out = total_trips_with_odo - trips_with_distance
                    st.info(f"ℹ️ **Đã lọc bỏ {filtered_out} chuyến có dữ liệu bất thường** (ODO giảm hoặc quãng đường > 2000km)")
            else:
                st.info("Không có dữ liệu quãng đường để hiển thị")
    
    # Tiếp tục phần code gốc với các chỉ số tần suất hoạt động
    st.header("🎯 Chỉ số tần suất hoạt động")
    
    # Tính toán các chỉ số
    total_days = (filtered_df["Ngày ghi nhận"].max() - filtered_df["Ngày ghi nhận"].min()).days + 1 if not filtered_df.empty else 1
    active_days_per_vehicle = filtered_df.groupby("Mã xe")["Ngày"].nunique()
    trips_per_day_per_vehicle = filtered_df.groupby("Mã xe").size() / active_days_per_vehicle
    utilization_rate = (active_days_per_vehicle / total_days * 100)
    
    # Phân tích tài xế
    if "Tên tài xế" in filtered_df.columns:
        driver_stats = filtered_df.groupby("Tên tài xế").agg({
            "Mã xe": "nunique",
            "STT": "count",
            "Doanh thu": "sum",
            "Thời gian chạy (giờ)": "sum",
            "Ngày": "nunique"
        }).rename(columns={
            "Mã xe": "Số xe điều khiển",
            "STT": "Số chuyến",
            "Doanh thu": "Tổng doanh thu",
            "Thời gian chạy (giờ)": "Tổng giờ lái",
            "Ngày": "Số ngày làm việc"
        })
    else:
        driver_stats = filtered_df.groupby("Email").agg({
            "Mã xe": "nunique",
            "STT": "count",
            "Doanh thu": "sum",
            "Thời gian chạy (giờ)": "sum",
            "Ngày": "nunique"
        }).rename(columns={
            "Mã xe": "Số xe điều khiển",
            "STT": "Số chuyến",
            "Doanh thu": "Tổng doanh thu",
            "Thời gian chạy (giờ)": "Tổng giờ lái",
            "Ngày": "Số ngày làm việc"
        })
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_trips_per_day = len(filtered_df) / total_days if total_days > 0 else 0
        st.metric(
            label="Chuyến/ngày trung bình",
            value=f"{avg_trips_per_day:.1f}",
            help="Số chuyến trung bình mỗi ngày của toàn bộ đội xe"
        )
    
    with col2:
        avg_utilization = utilization_rate.mean() if not utilization_rate.empty else 0
        st.metric(
            label="Tỷ lệ sử dụng xe TB",
            value=f"{avg_utilization:.1f}%",
            help="Tỷ lệ sử dụng trung bình của các xe (ngày hoạt động/tổng ngày)"
        )
    
    with col3:
        peak_day_trips = filtered_df.groupby("Ngày").size().max() if not filtered_df.empty else 0
        st.metric(
            label="Ngày cao điểm",
            value=f"{peak_day_trips} chuyến",
            help="Số chuyến cao nhất trong 1 ngày"
        )
    
    # Bảng chi tiết hiệu suất xe (TÍCH HỢP QUÃNG ĐƯỜNG)
    st.subheader("📋 Hiệu suất chi tiết từng xe")
    
    vehicle_performance = filtered_df.groupby("Mã xe").agg({
        "STT": "count",
        "Doanh thu": ["sum", "mean"],
        "Thời gian chạy (giờ)": "sum",
        "Ngày": "nunique"
    }).round(2)
    
    # Flatten column names
    vehicle_performance.columns = [
        "Tổng chuyến", "Tổng doanh thu", "Doanh thu TB/chuyến", 
        "Tổng giờ chạy", "Số ngày hoạt động"
    ]
    
    # Thêm quãng đường nếu có
    if "Quãng đường (km)" in filtered_df.columns:
        distance_stats = filtered_df.groupby("Mã xe")["Quãng đường (km)"].agg(['sum', 'mean', 'count']).round(1)
        distance_stats.columns = ["Tổng QĐ (km)", "QĐ TB (km)", "Chuyến có QĐ"]
        vehicle_performance = vehicle_performance.join(distance_stats, how='left')
        
        # Tính tốc độ trung bình
        vehicle_performance["Tốc độ TB (km/h)"] = (
            vehicle_performance["Tổng QĐ (km)"] / vehicle_performance["Tổng giờ chạy"]
        ).round(1)
    
    # Thêm các chỉ số tính toán
    vehicle_performance["Chuyến/ngày"] = (vehicle_performance["Tổng chuyến"] / vehicle_performance["Số ngày hoạt động"]).round(1)
    vehicle_performance["Tỷ lệ sử dụng (%)"] = (vehicle_performance["Số ngày hoạt động"] / total_days * 100).round(1)
    vehicle_performance["Hiệu suất"] = vehicle_performance.apply(
        lambda row: "Cao" if row["Chuyến/ngày"] >= 2 and row["Tỷ lệ sử dụng (%)"] >= 70 
        else "Trung bình" if row["Chuyến/ngày"] >= 1 and row["Tỷ lệ sử dụng (%)"] >= 50
        else "Thấp", axis=1
    )
    
    # Sắp xếp theo hiệu suất
    vehicle_performance = vehicle_performance.sort_values("Tổng doanh thu", ascending=False)
    
    # Format cho hiển thị
    format_dict = {
        "Tổng doanh thu": "{:,.0f}",
        "Doanh thu TB/chuyến": "{:,.0f}",
        "Tổng giờ chạy": "{:.1f}",
        "Chuyến/ngày": "{:.1f}",
        "Tỷ lệ sử dụng (%)": "{:.1f}"
    }
    
    if "Tổng QĐ (km)" in vehicle_performance.columns:
        format_dict.update({
            "Tổng QĐ (km)": "{:.1f}",
            "QĐ TB (km)": "{:.1f}",
            "Tốc độ TB (km/h)": "{:.1f}"
        })
    
    # Sử dụng st.data_editor thay vì st.dataframe để tránh lỗi PyArrow
    st.data_editor(
        vehicle_performance.style.format(format_dict),
        use_container_width=True,
        disabled=True,
        hide_index=False
    )
    
    # Phân tích tài xế (TÍCH HỢP QUÃNG ĐƯỜNG)
    st.subheader("👨‍💼 Hiệu suất tài xế")
    
    driver_stats["Chuyến/ngày"] = (driver_stats["Số chuyến"] / driver_stats["Số ngày làm việc"]).round(1)
    driver_stats["Giờ/ngày"] = (driver_stats["Tổng giờ lái"] / driver_stats["Số ngày làm việc"]).round(1)
    
    # Thêm thống kê quãng đường cho tài xế
    if "Quãng đường (km)" in filtered_df.columns and "Tên tài xế" in filtered_df.columns:
        driver_distance = filtered_df.groupby("Tên tài xế")["Quãng đường (km)"].agg(['sum', 'mean', 'count']).round(1)
        driver_distance.columns = ["Tổng QĐ (km)", "QĐ TB (km)", "Chuyến có QĐ"]
        driver_stats = driver_stats.join(driver_distance, how='left')
    
    driver_stats = driver_stats.sort_values("Tổng doanh thu", ascending=False)
    
    driver_format_dict = {
        "Tổng doanh thu": "{:,.0f}",
        "Tổng giờ lái": "{:.1f}",
        "Chuyến/ngày": "{:.1f}",
        "Giờ/ngày": "{:.1f}"
    }
    
    if "Tổng QĐ (km)" in driver_stats.columns:
        driver_format_dict.update({
            "Tổng QĐ (km)": "{:.1f}",
            "QĐ TB (km)": "{:.1f}"
        })
    
    # Sử dụng st.data_editor thay vì st.dataframe
    st.data_editor(
        driver_stats.style.format(driver_format_dict),
        use_container_width=True,
        disabled=True,
        hide_index=False
    )
    
    # Phân tích theo loại xe
    st.header("🚗 Phân tích theo loại xe")
    
    admin_vehicles_list = ["50A-004.55", "50A-007.20", "50A-012.59", "51B-330.67"]
    
    df_with_type = filtered_df.copy()
    df_with_type["Loại xe"] = df_with_type["Mã xe"].apply(
        lambda x: "🏢 Xe hành chính" if x in admin_vehicles_list else "🏥 Xe cứu thương"
    )
    
    # Hiển thị danh sách xe
    st.subheader("📝 Danh sách phân loại xe")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏢 Xe hành chính:**")
        for vehicle in admin_vehicles_list:
            if vehicle in filtered_df["Mã xe"].values:
                st.write(f"✅ {vehicle}")
            else:
                st.write(f"❌ {vehicle} (không có trong dữ liệu được lọc)")
    
    with col2:
        st.write("**🏥 Xe cứu thương:**")
        ambulance_vehicles_in_data = [v for v in filtered_df["Mã xe"].unique() if v not in admin_vehicles_list]
        if ambulance_vehicles_in_data:
            for vehicle in sorted(ambulance_vehicles_in_data):
                st.write(f"✅ {vehicle}")
        else:
            st.write("Không có xe cứu thương trong dữ liệu được lọc")
    
    # Tính toán thống kê theo loại xe (TÍCH HỢP QUÃNG ĐƯỜNG)
    vehicle_type_stats = df_with_type.groupby("Loại xe").agg({
        "Mã xe": "nunique",
        "STT": "count", 
        "Doanh thu": "sum",
        "Thời gian chạy (giờ)": "sum"
    }).rename(columns={
        "Mã xe": "Số xe",
        "STT": "Tổng chuyến",
        "Doanh thu": "Tổng doanh thu",
        "Thời gian chạy (giờ)": "Tổng giờ chạy"
    })
    
    # Thêm thống kê quãng đường theo loại xe
    if "Quãng đường (km)" in df_with_type.columns:
        distance_by_type = df_with_type.groupby("Loại xe")["Quãng đường (km)"].agg(['sum', 'mean']).round(1)
        distance_by_type.columns = ["Tổng QĐ (km)", "QĐ TB (km)"]
        vehicle_type_stats = vehicle_type_stats.join(distance_by_type, how='left')
    
    # Hiển thị metrics tổng quan theo loại xe
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Xe cứu thương")
        ambulance_data = vehicle_type_stats.loc["🏥 Xe cứu thương"] if "🏥 Xe cứu thương" in vehicle_type_stats.index else None
        if ambulance_data is not None:
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Số xe", f"{ambulance_data['Số xe']}")
            with col_a2:
                st.metric("Tổng chuyến", f"{ambulance_data['Tổng chuyến']:,}")
            with col_a3:
                st.metric("Doanh thu", f"{ambulance_data['Tổng doanh thu']:,.0f} VNĐ")
            
            if "Tổng QĐ (km)" in ambulance_data.index:
                col_a4, col_a5 = st.columns(2)
                with col_a4:
                    st.metric("Tổng QĐ", f"{ambulance_data['Tổng QĐ (km)']:.1f} km")
                with col_a5:
                    st.metric("QĐ TB", f"{ambulance_data['QĐ TB (km)']:.1f} km")
        else:
            st.info("Không có xe cứu thương trong dữ liệu")
    
    with col2:
        st.subheader("🏢 Xe hành chính")
        admin_data = vehicle_type_stats.loc["🏢 Xe hành chính"] if "🏢 Xe hành chính" in vehicle_type_stats.index else None
        if admin_data is not None:
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Số xe", f"{admin_data['Số xe']}")
            with col_b2:
                st.metric("Tổng chuyến", f"{admin_data['Tổng chuyến']:,}")
            with col_b3:
                st.metric("Tổng giờ chạy", f"{admin_data['Tổng giờ chạy']:.1f} giờ")
            
            if "Tổng QĐ (km)" in admin_data.index:
                col_b4, col_b5 = st.columns(2)
                with col_b4:
                    st.metric("Tổng QĐ", f"{admin_data['Tổng QĐ (km)']:.1f} km")
                with col_b5:
                    st.metric("QĐ TB", f"{admin_data['QĐ TB (km)']:.1f} km")
        else:
            st.info("Không có xe hành chính trong dữ liệu")
    
    # Biểu đồ chính
    st.header("📈 Phân tích chi tiết")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Doanh thu", "🚗 Hiệu suất xe", "📅 Theo thời gian", 
        "🎯 Loại công tác", "🔍 Phân tích loại xe", "⚠️ Phân tích quá tải", "🛣️ Phân tích quãng đường"
    ])
    
    with tab7:
        # TAB MỚI: PHÂN TÍCH QUÃNG ĐƯỜNG CHI TIẾT
        if "Quãng đường (km)" in filtered_df.columns:
            st.subheader("🛣️ Phân tích chi tiết quãng đường")
            
            distance_df = filtered_df[filtered_df["Quãng đường (km)"].notna()]
            
            if not distance_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Phân bố quãng đường (lọc bỏ outliers > 1000km để biểu đồ rõ hơn)
                    reasonable_distance = distance_df[distance_df["Quãng đường (km)"] <= 1000]
                    if not reasonable_distance.empty:
                        fig_dist_histogram = px.histogram(
                            reasonable_distance,
                            x="Quãng đường (km)",
                            title="Phân bố quãng đường (≤1000km)",
                            labels={"Quãng đường (km)": "Quãng đường (km)", "count": "Số chuyến"},
                            nbins=20
                        )
                        fig_dist_histogram.update_layout(height=400)
                        st.plotly_chart(fig_dist_histogram, use_container_width=True)
                        
                        # Hiển thị thông tin về outliers
                        outliers = distance_df[distance_df["Quãng đường (km)"] > 1000]
                        if not outliers.empty:
                            st.caption(f"📊 Biểu đồ hiển thị {len(reasonable_distance)} chuyến ≤1000km. Có {len(outliers)} chuyến >1000km không hiển thị.")
                    else:
                        st.info("Không có dữ liệu quãng đường hợp lệ để hiển thị histogram")
                
                with col2:
                    # Box plot quãng đường theo xe (cũng lọc outliers)
                    if not reasonable_distance.empty:
                        fig_box = px.box(
                            reasonable_distance,
                            x="Mã xe",
                            y="Quãng đường (km)",
                            title="Phân bố quãng đường theo xe (≤1000km)",
                            labels={"Quãng đường (km)": "Quãng đường (km)", "Mã xe": "Mã xe"}
                        )
                        fig_box.update_layout(height=400)
                        fig_box.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info("Không có dữ liệu quãng đường hợp lệ để hiển thị box plot")
                
                # Scatter plot: Quãng đường vs Doanh thu
                st.subheader("📊 Mối quan hệ quãng đường - Doanh thu - Thời gian")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_scatter1 = px.scatter(
                        distance_df,
                        x="Quãng đường (km)",
                        y="Doanh thu",
                        color="Mã xe",
                        title="Quãng đường vs Doanh thu",
                        labels={"Quãng đường (km)": "Quãng đường (km)", "Doanh thu": "Doanh thu (VNĐ)"},
                        hover_data=["Điểm đến", "Phân loại công tác"]
                    )
                    fig_scatter1.update_layout(height=400)
                    st.plotly_chart(fig_scatter1, use_container_width=True)
                
                with col2:
                    fig_scatter2 = px.scatter(
                        distance_df,
                        x="Quãng đường (km)",
                        y="Thời gian chạy (giờ)",
                        color="Mã xe",
                        title="Quãng đường vs Thời gian chạy",
                        labels={"Quãng đường (km)": "Quãng đường (km)", "Thời gian chạy (giờ)": "Thời gian (giờ)"},
                        hover_data=["Điểm đến", "Doanh thu"]
                    )
                    fig_scatter2.update_layout(height=400)
                    st.plotly_chart(fig_scatter2, use_container_width=True)
                
                # Top destinations by distance
                st.subheader("📍 Điểm đến xa nhất")
                
                destination_stats = distance_df.groupby("Điểm đến").agg({
                    "Quãng đường (km)": ["mean", "max", "count"],
                    "Doanh thu": "mean"
                }).round(1)
                
                destination_stats.columns = ["QĐ TB (km)", "QĐ Max (km)", "Số chuyến", "Doanh thu TB"]
                destination_stats = destination_stats.sort_values("QĐ TB (km)", ascending=False).head(10)
                
                # Tạo bản copy và format cho hiển thị
                display_dest = destination_stats.copy()
                display_dest = display_dest.reset_index()  # Đưa "Điểm đến" thành cột thường
                
                st.data_editor(
                    display_dest.style.format({
                        "QĐ TB (km)": "{:.1f}",
                        "QĐ Max (km)": "{:.1f}",
                        "Doanh thu TB": "{:,.0f}"
                    }),
                    use_container_width=True,
                    disabled=True,
                    hide_index=True
                )
            else:
                st.info("Không có dữ liệu quãng đường để phân tích chi tiết")
        else:
            st.warning("⚠️ Chưa có dữ liệu quãng đường. Hãy kiểm tra lại file dữ liệu.")
    
    # Các tab còn lại giữ nguyên như code gốc
    with tab6:
        st.subheader("🚨 Phân tích quá tải xe theo loại")
        
        filtered_df_overload = filtered_df.copy()
        filtered_df_overload["Loại xe"] = filtered_df_overload["Mã xe"].apply(
            lambda x: "🏢 Xe hành chính" if x in admin_vehicles_list else "🏥 Xe cứu thương"
        )
        
        total_admin_vehicles = len([v for v in df["Mã xe"].unique() if v in admin_vehicles_list])
        total_ambulance_vehicles = len([v for v in df["Mã xe"].unique() if v not in admin_vehicles_list])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏢 Phân tích quá tải xe hành chính")
            
            admin_data = filtered_df_overload[filtered_df_overload["Loại xe"] == "🏢 Xe hành chính"]
            
            if not admin_data.empty:
                admin_daily_analysis = admin_data.groupby("Ngày").agg({
                    "Mã xe": ["count", "nunique"],
                    "STT": "count"
                }).round(2)
                
                admin_daily_analysis.columns = ["Số chuyến", "Số xe sử dụng", "Tổng chuyến"]
                admin_daily_analysis = admin_daily_analysis.drop("Tổng chuyến", axis=1)
                
                admin_daily_analysis["Tỷ lệ sử dụng xe (%)"] = (admin_daily_analysis["Số xe sử dụng"] / total_admin_vehicles * 100).round(1)
                admin_daily_analysis["Tình trạng"] = admin_daily_analysis["Tỷ lệ sử dụng xe (%)"].apply(
                    lambda x: "🔴 Quá tải" if x >= 80 else "🟡 Cao" if x >= 60 else "🟢 Bình thường"
                )
                
                admin_daily_chart_data = admin_daily_analysis.reset_index()
                fig_admin_day = px.line(
                    admin_daily_chart_data,
                    x="Ngày",
                    y="Tỷ lệ sử dụng xe (%)",
                    title="Tỷ lệ sử dụng xe hành chính theo ngày",
                    labels={"Ngày": "Ngày", "Tỷ lệ sử dụng xe (%)": "Tỷ lệ sử dụng xe (%)"},
                    markers=True,
                    color_discrete_sequence=["#4ecdc4"]
                )
                fig_admin_day.add_hline(y=80, line_dash="dash", line_color="red", 
                                       annotation_text="Ngưỡng quá tải (80%)")
                fig_admin_day.add_hline(y=60, line_dash="dash", line_color="orange", 
                                       annotation_text="Ngưỡng cao (60%)")
                fig_admin_day.update_layout(height=400)
                st.plotly_chart(fig_admin_day, use_container_width=True)
                
                admin_overloaded_days = admin_daily_analysis[admin_daily_analysis["Tỷ lệ sử dụng xe (%)"] >= 80]
                admin_high_load_days = admin_daily_analysis[admin_daily_analysis["Tỷ lệ sử dụng xe (%)"] >= 60]
                admin_max_usage = admin_daily_analysis["Tỷ lệ sử dụng xe (%)"].max() if not admin_daily_analysis.empty else 0
                
                col_a1, col_a2, col_a3 = st.columns(3)
                
                with col_a1:
                    st.metric(
                        label="📊 Tổng xe hành chính",
                        value=f"{total_admin_vehicles} xe",
                        help="Tổng số xe hành chính có sẵn"
                    )
                
                with col_a2:
                    st.metric(
                        label="⚠️ Ngày quá tải",
                        value=f"{len(admin_overloaded_days)} ngày",
                        delta="≥80% xe đang sử dụng" if len(admin_overloaded_days) > 0 else "Không có",
                        delta_color="inverse" if len(admin_overloaded_days) > 0 else "normal"
                    )
                
                with col_a3:
                    st.metric(
                        label="📈 Tỷ lệ sử dụng cao nhất",
                        value=f"{admin_max_usage:.1f}%",
                        help="Tỷ lệ sử dụng xe hành chính cao nhất trong 1 ngày"
                    )
                
                if not admin_overloaded_days.empty:
                    st.error(f"🚨 **{len(admin_overloaded_days)} ngày quá tải xe hành chính** (≥80% xe đang sử dụng)")
                    st.data_editor(
                        admin_overloaded_days.sort_values("Tỷ lệ sử dụng xe (%)", ascending=False).head(10),
                        use_container_width=True,
                        disabled=True,
                        hide_index=False
                    )
                elif not admin_high_load_days.empty:
                    st.warning(f"⚠️ **{len(admin_high_load_days)} ngày tải cao xe hành chính** (≥60% xe đang sử dụng)")
                    st.data_editor(
                        admin_high_load_days.sort_values("Tỷ lệ sử dụng xe (%)", ascending=False).head(5),
                        use_container_width=True,
                        disabled=True,
                        hide_index=False
                    )
                else:
                    st.success("✅ **Không có ngày nào quá tải xe hành chính**")
            else:
                st.info("Không có dữ liệu xe hành chính trong khoảng thời gian được chọn")
        
        with col2:
            st.subheader("🏥 Phân tích quá tải xe cứu thương")
            
            ambulance_data = filtered_df_overload[filtered_df_overload["Loại xe"] == "🏥 Xe cứu thương"]
            
            if not ambulance_data.empty:
                ambulance_daily_analysis = ambulance_data.groupby("Ngày").agg({
                    "Mã xe": ["count", "nunique"],
                    "STT": "count"
                }).round(2)
                
                ambulance_daily_analysis.columns = ["Số chuyến", "Số xe sử dụng", "Tổng chuyến"]
                ambulance_daily_analysis = ambulance_daily_analysis.drop("Tổng chuyến", axis=1)
                
                ambulance_daily_analysis["Tỷ lệ sử dụng xe (%)"] = (ambulance_daily_analysis["Số xe sử dụng"] / total_ambulance_vehicles * 100).round(1)
                ambulance_daily_analysis["Tình trạng"] = ambulance_daily_analysis["Tỷ lệ sử dụng xe (%)"].apply(
                    lambda x: "🔴 Quá tải" if x >= 80 else "🟡 Cao" if x >= 60 else "🟢 Bình thường"
                )
                
                ambulance_daily_chart_data = ambulance_daily_analysis.reset_index()
                fig_ambulance_day = px.line(
                    ambulance_daily_chart_data,
                    x="Ngày",
                    y="Tỷ lệ sử dụng xe (%)",
                    title="Tỷ lệ sử dụng xe cứu thương theo ngày",
                    labels={"Ngày": "Ngày", "Tỷ lệ sử dụng xe (%)": "Tỷ lệ sử dụng xe (%)"},
                    markers=True,
                    color_discrete_sequence=["#ff6b6b"]
                )
                fig_ambulance_day.add_hline(y=80, line_dash="dash", line_color="red", 
                                           annotation_text="Ngưỡng quá tải (80%)")
                fig_ambulance_day.add_hline(y=60, line_dash="dash", line_color="orange", 
                                           annotation_text="Ngưỡng cao (60%)")
                fig_ambulance_day.update_layout(height=400)
                st.plotly_chart(fig_ambulance_day, use_container_width=True)
                
                ambulance_overloaded_days = ambulance_daily_analysis[ambulance_daily_analysis["Tỷ lệ sử dụng xe (%)"] >= 80]
                ambulance_high_load_days = ambulance_daily_analysis[ambulance_daily_analysis["Tỷ lệ sử dụng xe (%)"] >= 60]
                ambulance_max_usage = ambulance_daily_analysis["Tỷ lệ sử dụng xe (%)"].max() if not ambulance_daily_analysis.empty else 0
                
                col_b1, col_b2, col_b3 = st.columns(3)
                
                with col_b1:
                    st.metric(
                        label="📊 Tổng xe cứu thương",
                        value=f"{total_ambulance_vehicles} xe",
                        help="Tổng số xe cứu thương có sẵn"
                    )
                
                with col_b2:
                    st.metric(
                        label="⚠️ Ngày quá tải",
                        value=f"{len(ambulance_overloaded_days)} ngày",
                        delta="≥80% xe đang sử dụng" if len(ambulance_overloaded_days) > 0 else "Không có",
                        delta_color="inverse" if len(ambulance_overloaded_days) > 0 else "normal"
                    )
                
                with col_b3:
                    st.metric(
                        label="📈 Tỷ lệ sử dụng cao nhất",
                        value=f"{ambulance_max_usage:.1f}%",
                        help="Tỷ lệ sử dụng xe cứu thương cao nhất trong 1 ngày"
                    )
                
                if not ambulance_overloaded_days.empty:
                    st.error(f"🚨 **{len(ambulance_overloaded_days)} ngày quá tải xe cứu thương** (≥80% xe đang sử dụng)")
                    st.data_editor(
                        ambulance_overloaded_days.sort_values("Tỷ lệ sử dụng xe (%)", ascending=False).head(10),
                        use_container_width=True,
                        disabled=True,
                        hide_index=False
                    )
                elif not ambulance_high_load_days.empty:
                    st.warning(f"⚠️ **{len(ambulance_high_load_days)} ngày tải cao xe cứu thương** (≥60% xe đang sử dụng)")
                    st.data_editor(
                        ambulance_high_load_days.sort_values("Tỷ lệ sử dụng xe (%)", ascending=False).head(5),
                        use_container_width=True,
                        disabled=True,
                        hide_index=False
                    )
                else:
                    st.success("✅ **Không có ngày nào quá tải xe cứu thương**")
            else:
                st.info("Không có dữ liệu xe cứu thương trong khoảng thời gian được chọn")
    
    # Các tab khác
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_by_vehicle = filtered_df.groupby("Mã xe")["Doanh thu"].sum().sort_values(ascending=False)
            fig1 = px.bar(
                x=revenue_by_vehicle.index,
                y=revenue_by_vehicle.values,
                title="Doanh thu theo xe",
                labels={"x": "Mã xe", "y": "Doanh thu (VNĐ)"},
                color=revenue_by_vehicle.values,
                color_continuous_scale="Blues"
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            top_trips = filtered_df.nlargest(10, "Doanh thu")[["Mã xe", "Doanh thu", "Điểm đến", "Phân loại công tác"]]
            fig2 = px.bar(
                top_trips,
                x="Doanh thu",
                y="Mã xe",
                title="Top 10 chuyến có doanh thu cao nhất",
                orientation="h",
                color="Doanh thu",
                color_continuous_scale="Greens"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            trips_by_vehicle = filtered_df["Mã xe"].value_counts()
            fig3 = px.pie(
                values=trips_by_vehicle.values,
                names=trips_by_vehicle.index,
                title="Phân bố số chuyến theo xe"
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            hours_by_vehicle = filtered_df.groupby("Mã xe")["Thời gian chạy (giờ)"].sum().sort_values(ascending=False)
            fig4 = px.bar(
                x=hours_by_vehicle.index,
                y=hours_by_vehicle.values,
                title="Tổng giờ chạy theo xe",
                labels={"x": "Mã xe", "y": "Thời gian (giờ)"},
                color=hours_by_vehicle.values,
                color_continuous_scale="Reds"
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if "Tháng" in filtered_df.columns:
                monthly_revenue = filtered_df.groupby("Tháng")["Doanh thu"].sum()
                fig5 = px.line(
                    x=monthly_revenue.index,
                    y=monthly_revenue.values,
                    title="Xu hướng doanh thu theo tháng",
                    labels={"x": "Tháng", "y": "Doanh thu (VNĐ)"},
                    markers=True
                )
                fig5.update_layout(height=400)
                st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            if "Tháng" in filtered_df.columns:
                monthly_trips = filtered_df.groupby("Tháng").size()
                fig6 = px.bar(
                    x=monthly_trips.index,
                    y=monthly_trips.values,
                    title="Số chuyến theo tháng",
                    labels={"x": "Tháng", "y": "Số chuyến"},
                    color=monthly_trips.values,
                    color_continuous_scale="Purples"
                )
                fig6.update_layout(height=400)
                st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            work_type_dist = filtered_df["Phân loại công tác"].dropna().value_counts()
            if not work_type_dist.empty:
                fig7 = px.pie(
                    values=work_type_dist.values,
                    names=work_type_dist.index,
                    title="Phân bố theo loại công tác"
                )
                fig7.update_layout(height=400)
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("Không có dữ liệu loại công tác")
        
        with col2:
            revenue_by_work_type = filtered_df.groupby("Phân loại công tác")["Doanh thu"].sum().sort_values(ascending=False)
            revenue_by_work_type = revenue_by_work_type.dropna()
            if not revenue_by_work_type.empty:
                fig8 = px.bar(
                    x=revenue_by_work_type.values,
                    y=revenue_by_work_type.index,
                    title="Doanh thu theo loại công tác",
                    orientation="h",
                    labels={"x": "Doanh thu (VNĐ)", "y": "Loại công tác"},
                    color=revenue_by_work_type.values,
                    color_continuous_scale="Oranges"
                )
                fig8.update_layout(height=400)
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("Không có dữ liệu doanh thu theo loại công tác")
    
    with tab5:
        st.subheader("🔍 Phân tích chi tiết theo loại xe")
        
        filtered_with_type = filtered_df.copy()
        filtered_with_type["Loại xe"] = filtered_with_type["Mã xe"].apply(
            lambda x: "🏢 Xe hành chính" if x in admin_vehicles_list else "🏥 Xe cứu thương"
        )
        
        if "Tháng" in filtered_with_type.columns:
            monthly_by_type = filtered_with_type.groupby(["Tháng", "Loại xe"]).agg({
                "STT": "count",
                "Doanh thu": "sum"
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_monthly_trips = px.line(
                    monthly_by_type,
                    x="Tháng",
                    y="STT",
                    color="Loại xe",
                    title="Xu hướng số chuyến theo tháng và loại xe",
                    labels={"STT": "Số chuyến", "Tháng": "Tháng"},
                    color_discrete_map={
                        "🏥 Xe cứu thương": "#ff6b6b",
                        "🏢 Xe hành chính": "#4ecdc4"
                    },
                    markers=True
                )
                fig_monthly_trips.update_layout(height=400)
                st.plotly_chart(fig_monthly_trips, use_container_width=True)
            
            with col2:
                ambulance_monthly = monthly_by_type[monthly_by_type["Loại xe"] == "🏥 Xe cứu thương"]
                if not ambulance_monthly.empty:
                    fig_monthly_revenue = px.bar(
                        ambulance_monthly,
                        x="Tháng",
                        y="Doanh thu",
                        title="Doanh thu xe cứu thương theo tháng",
                        labels={"Doanh thu": "Doanh thu (VNĐ)", "Tháng": "Tháng"},
                        color_discrete_sequence=["#ff6b6b"]
                    )
                    fig_monthly_revenue.update_layout(height=400)
                    st.plotly_chart(fig_monthly_revenue, use_container_width=True)
                else:
                    st.info("Không có dữ liệu doanh thu xe cứu thương theo tháng")
    
    # Bảng chi tiết
    st.header("📋 Dữ liệu chi tiết")
    
    default_columns = ["Mã xe", "Ngày ghi nhận", "Điểm đến", "Phân loại công tác", "Doanh thu", "Thời gian chạy (phút)"]
    if "Tên tài xế" in filtered_df.columns:
        default_columns.insert(1, "Tên tài xế")
    if "Quãng đường (km)" in filtered_df.columns:
        default_columns.append("Quãng đường (km)")
    
    show_columns = st.multiselect(
        "Chọn cột hiển thị:",
        options=filtered_df.columns.tolist(),
        default=default_columns
    )
    
    if show_columns:
        # Tạo bản sao để tránh lỗi SettingWithCopyWarning
        display_df = filtered_df[show_columns].copy().sort_values("Ngày ghi nhận", ascending=False)
        
        # Xử lý cột quãng đường để tránh lỗi PyArrow
        if "Quãng đường (km)" in show_columns and "Quãng đường (km)" in display_df.columns:
            # Chuyển đổi sang string để tránh lỗi PyArrow với mixed types
            display_df["Quãng đường (km)"] = display_df["Quãng đường (km)"].apply(
                lambda x: f"{x:.1f} km" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
        
        # Sử dụng st.data_editor thay vì st.dataframe để tránh lỗi PyArrow
        st.data_editor(
            display_df,
            use_container_width=True,
            height=400,
            disabled=True,  # Chỉ đọc
            hide_index=True
        )
    
    # Xuất báo cáo (TÍCH HỢP QUÃNG ĐƯỜNG)
    st.header("📄 Báo cáo giao ban")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Tạo báo cáo giao ban", type="primary"):
            total_days = (filtered_df["Ngày ghi nhận"].max() - filtered_df["Ngày ghi nhận"].min()).days + 1 if not filtered_df.empty else 1
            avg_trips_per_day = len(filtered_df) / total_days if total_days > 0 else 0
            max_capacity = len(df["Mã xe"].unique()) * 2
            
            high_perf_count = len(vehicle_performance[vehicle_performance["Hiệu suất"] == "Cao"])
            low_perf_count = len(vehicle_performance[vehicle_performance["Hiệu suất"] == "Thấp"])
            
            need_external = "CẦN THUÊ XE BÊN NGOÀI" if avg_trips_per_day > max_capacity * 0.8 else "HIỆN TẠI ĐỦ NĂNG LỰC"
            
            report_data = {
                "🚗 BÁO CÁO HOẠT ĐỘNG TỔ XE": {
                    "Thời gian báo cáo": f"{filtered_df['Ngày ghi nhận'].min().strftime('%d/%m/%Y')} - {filtered_df['Ngày ghi nhận'].max().strftime('%d/%m/%Y')}",
                    "Tổng số chuyến": f"{len(filtered_df):,} chuyến",
                    "Trung bình chuyến/ngày": f"{avg_trips_per_day:.1f} chuyến",
                    "Tổng doanh thu": f"{total_revenue:,.0f} VNĐ",
                    "Số xe hoạt động": f"{num_vehicles}/{df['Mã xe'].nunique()} xe"
                },
                "📊 TẦN SUẤT HOẠT ĐỘNG": {
                    f"Xe hiệu suất cao": f"{high_perf_count} xe",
                    f"Xe hiệu suất thấp": f"{low_perf_count} xe", 
                    "Tỷ lệ sử dụng trung bình": f"{utilization_rate.mean():.1f}%",
                    "Ngày cao điểm": f"{peak_day_trips} chuyến"
                },
                "💼 HIỆU SUẤT TÀI XẾ": {
                    "Số tài xế hoạt động": f"{len(driver_stats)} người",
                    "Tài xế tích cực (>1.5 chuyến/ngày)": f"{len(driver_stats[driver_stats['Chuyến/ngày'] >= 1.5])} người",
                    "Tài xế ít hoạt động (<1 chuyến/ngày)": f"{len(driver_stats[driver_stats['Chuyến/ngày'] < 1])} người"
                }
            }
            
            # Thêm phần quãng đường vào báo cáo
            if "Quãng đường (km)" in filtered_df.columns:
                distance_data = filtered_df[filtered_df["Quãng đường (km)"].notna()]
                if not distance_data.empty:
                    total_distance = distance_data["Quãng đường (km)"].sum()
                    avg_distance = distance_data["Quãng đường (km)"].mean()
                    coverage = len(distance_data) / len(filtered_df) * 100
                    
                    report_data["🛣️ QUÃNG ĐƯỜNG DI CHUYỂN"] = {
                        "Tổng quãng đường": f"{total_distance:.1f} km",
                        "Quãng đường TB/chuyến": f"{avg_distance:.1f} km",
                        "Tỷ lệ tính được quãng đường": f"{coverage:.1f}%",
                        "Số chuyến có quãng đường": f"{len(distance_data)}/{len(filtered_df)} chuyến"
                    }
                    
                    if total_hours > 0:
                        avg_speed = total_distance / total_hours
                        report_data["🛣️ QUÃNG ĐƯỜNG DI CHUYỂN"]["Tốc độ trung bình"] = f"{avg_speed:.1f} km/h"
            
            report_data["🎯 KHUYẾN NGHỊ"] = {
                "Đánh giá năng lực": need_external,
                "Công suất hiện tại": f"{(avg_trips_per_day/max_capacity)*100:.1f}% công suất tối đa",
                "Đề xuất": "Tối ưu hóa xe hiệu suất thấp, đào tạo tài xế ít hoạt động" if avg_trips_per_day <= max_capacity * 0.8 else "Cần thuê xe bên ngoài cho ngày cao điểm"
            }
            
            st.json(report_data)
    
    with col2:
        # Tải dữ liệu đã lọc
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Tải dữ liệu chi tiết (CSV)",
            data=csv,
            file_name=f"bao_cao_xe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Tải báo cáo hiệu suất xe
        vehicle_csv = vehicle_performance.to_csv(encoding='utf-8-sig')
        st.download_button(
            label="📊 Tải báo cáo hiệu suất xe",
            data=vehicle_csv,
            file_name=f"hieu_suat_xe_{datetime.now().strftime('%Y%m%d')}.csv", 
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

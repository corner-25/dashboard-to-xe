# 🚗 Dashboard Quản lý Đội xe Bệnh viện ĐHYD

Dashboard phân tích hiệu suất và quản lý đội xe với tính năng tự động tính toán quãng đường di chuyển.

## 👨‍💻 Tác giả
**Dương Hữu Quang**
- 📱 Contact: 0789106201 (Zalo/Telegram)
- 💼 LinkedIn: https://www.linkedin.com/in/huuquang-hcmut/
- 🏥 Thực tập sinh Phòng Hành Chính, Bệnh viện Đại học Y Dược (UMC)

## ✨ Tính năng chính

- 📊 **Tính toán quãng đường tự động** - Tự động tính quãng đường dựa trên chỉ số đồng hồ
- 📈 **Phân tích hiệu suất xe và tài xế** - Báo cáo chi tiết về hiệu suất hoạt động
- 🎯 **Metrics tổng quan** - Doanh thu, giờ chạy, quãng đường, tần suất sử dụng
- 📅 **Bộ lọc dữ liệu linh hoạt** - Lọc theo thời gian, xe, tài xế, loại công tác
- 📋 **Báo cáo giao ban** - Xuất báo cáo tự động cho cuộc họp
- 💾 **Export dữ liệu** - Tải về CSV cho phân tích nâng cao

## 🚀 Demo

**Live Demo:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## 📋 Yêu cầu

- Python 3.8+
- File Excel với sheet "Dữ liệu gộp"
- Các cột bắt buộc: STT, Mã xe, Email, Ngày ghi nhận, Chỉ số đồng hồ, Doanh thu, Thời gian chạy (phút)

## 🛠️ Cài đặt

1. **Clone repository**
```bash
git clone https://github.com/your-username/dashboard-doi-xe.git
cd dashboard-doi-xe
```

2. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

3. **Chạy ứng dụng**
```bash
streamlit run dashboard.py
```

## 📁 Cấu trúc project

```
dashboard-doi-xe/
├── dashboard.py              # Main application file
├── requirements.txt          # Python dependencies
├── Dashboard DHYD_ver24.xlsx  # Sample data file
├── assets/
│   └── logo.png             # Optional logo file
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # Documentation
```

## 📊 Hướng dẫn sử dụng

### 1. Upload dữ liệu
- Chọn file Excel (.xlsx/.xls) có sheet "Dữ liệu gộp"
- Hệ thống sẽ tự động xử lý và tính toán quãng đường

### 2. Sử dụng bộ lọc
- **Thời gian**: Lọc theo khoảng thời gian hoặc tháng/năm cụ thể
- **Xe**: Chọn xe muốn phân tích
- **Tài xế**: Lọc theo tài xế cụ thể
- **Loại công tác**: Phân loại theo mục đích sử dụng xe

### 3. Phân tích dữ liệu
- Xem metrics tổng quan ở đầu trang
- Phân tích quãng đường di chuyển chi tiết
- Đánh giá hiệu suất từng xe và tài xế
- Xem các biểu đồ phân tích theo tabs

### 4. Xuất báo cáo
- Tạo báo cáo giao ban tự động
- Tải dữ liệu chi tiết dạng CSV
- Export báo cáo hiệu suất xe

## 🔧 Cấu hình

### Streamlit Config (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
maxUploadSize = 200
```

### Logo
Đặt file `logo.png` vào thư mục `assets/` để hiển thị logo của bệnh viện.

## 📈 Tính năng tính toán quãng đường

Hệ thống tự động:
- Sắp xếp dữ liệu theo thời gian cho từng xe
- Tính quãng đường = Chỉ số đồng hồ hiện tại - Chỉ số đồng hồ trước đó
- Lọc bỏ dữ liệu bất thường (quãng đường âm hoặc > 640km)
- Hiển thị tỷ lệ coverage và phân tích chất lượng dữ liệu

## 🚀 Deploy lên Streamlit Cloud

1. **Push code lên GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Truy cập [share.streamlit.io](https://share.streamlit.io)**

3. **Connect GitHub repository**

4. **Cấu hình deploy**
   - Repository: `your-username/dashboard-doi-xe`
   - Branch: `main`
   - Main file path: `dashboard.py`

5. **Deploy** và đợi vài phút để ứng dụng khởi chạy

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

**Bệnh viện ĐHYD** - Tổ xe

Project Link: [https://github.com/your-username/dashboard-doi-xe](https://github.com/your-username/dashboard-doi-xe)

---

⭐ **Star repository này nếu nó hữu ích cho bạn!**
# Multimodal Sentiment Analysis on CMU-MOSEI

## 1. Giới thiệu

Repository này triển khai bài toán **phân tích cảm xúc đa phương thức** trên tập dữ liệu **CMU-MOSEI**. Thay vì chỉ sử dụng văn bản, mô hình khai thác đồng thời ba nguồn thông tin:

- **Text modality**: nội dung phát ngôn, được biểu diễn bằng embedding từ `bert-base-uncased`.
- **Audio modality**: đặc trưng âm thanh COVAREP, kích thước 74.
- **Visual modality**: đặc trưng khuôn mặt OpenFace 2.0, kích thước 47.

Mục tiêu của dự án là dự đoán điểm cảm xúc liên tục của mỗi đoạn video/phát ngôn và đánh giá mô hình bằng các thước đo như MAE, correlation, binary accuracy và binary F1-score.

## 2. Nội dung chính của repository

```text
.
├── Sentiment_analysis.ipynb      # Notebook chính: xử lý dữ liệu, xây dựng mô hình, huấn luyện và đánh giá
├── README.md                     # Mô tả dự án
└── scientific_paper_vi.md        # Bài báo khoa học mô tả phương pháp và kết quả thử nghiệm
```

## 3. Bài toán

Phân tích cảm xúc truyền thống thường chỉ dựa vào văn bản. Tuy nhiên, trong giao tiếp tự nhiên, cảm xúc còn được thể hiện thông qua giọng nói, ngữ điệu, nét mặt và cử chỉ. Vì vậy, bài toán trong repository này được thiết kế theo hướng **multimodal sentiment analysis**, kết hợp:

1. Nội dung ngôn ngữ.
2. Tín hiệu âm thanh.
3. Tín hiệu thị giác.

Đầu ra của mô hình là một giá trị cảm xúc liên tục. Khi cần đánh giá nhị phân, điểm cảm xúc được chuyển thành hai lớp:

- `positive`: sentiment score > 0
- `negative/non-positive`: sentiment score <= 0

## 4. Dataset

Dự án sử dụng tập dữ liệu **CMU-MOSEI** được tải thông qua `kagglehub`:

```python
path = kagglehub.dataset_download("samarwarsi/cmu-mosei")
```

Các file dữ liệu chính được sử dụng trong notebook:

```text
CMU_MOSEI_TimestampedWords.csd
CMU_MOSEI_COVAREP.csd
CMU_MOSEI_VisualOpenFace2.csd
CMU_MOSEI_Labels.csd
```

Trong lần chạy hiện tại của notebook, dữ liệu được đọc thành 3.293 segments và chia thành:

| Split | Số mẫu |
|---|---:|
| Train | 2.370 |
| Validation | 264 |
| Test | 658 |

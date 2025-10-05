# BÁO CÁO TIẾN ĐỘ DỰ ÁN VLM – NIH Chest X‑Ray (Stage 1)

---

## I. Tiền xử lý & Augmentation

### 1) Preprocessing

- **Lớp:** `CXRPreprocessor`
- **Chức năng chính:**
  - Resize/crop về kích thước chuẩn (`IMAGE_SIZE = 512`).
  - Tuỳ chọn **CLAHE** (tăng tương phản vùng phổi) – hiện đang **bật** mặc định trong demo.
  - Chuẩn hoá cường độ ảnh (normalize) theo thông số ImageNet khi cần.
  - Tuỳ chọn **lung mask** (đang **tắt** để đơn giản và nhanh hơn giai đoạn đầu).

### 2) Augmentations (Albumentations)

- **Train:** `HorizontalFlip (p=0.5)`, `ShiftScaleRotate` (dịch/scale/rotate nhẹ), `ElasticTransform` nhẹ, chuẩn hoá + `ToTensorV2()`.
- **Validation/Test:** chỉ **Normalize** + `ToTensorV2()`.

> Ghi chú: Augmentations được chọn **nhẹ** để tránh phá vỡ cấu trúc y khoa; sẽ tinh chỉnh sau khi có baseline.

---

## II. Dataset & Dataloader (Streaming GCS)

- **Lớp:** `GCSNIHCXRDataset(Dataset)`
- **Cơ chế đọc:** dùng `gcsfs` để mở tệp **PNG/JPG** trực tiếp trên `gs://nih-chest-xray/...` theo **từng batch**.
- **CSV dùng:** `Data_Entry_2017.csv` từ Drive (đã mount), lọc các dòng trỏ đến file ảnh tương ứng.
- **Bộ đệm (cache) tên file → thư mục:** giảm chi phí tra thư mục trên GCS lặp lại.
- **Tối ưu lần đầu:** lần đầu truy cập một filename sẽ chậm hơn; sau đó có cơ chế cache tên file và folder đích để tăng tốc.
- **Subset demo:** tuỳ chọn lấy 2% dữ liệu để thử nhanh (giảm thời gian I/O setup).

**Dataloader:**

- `batch_size = 8`, `num_workers = 2` (có thể tăng dần nếu ổn định).
- Kiểm thử `forward pass` với wrapper model đã **thành công** trên batch nhỏ.

---

## III. Model Wrappers

- **`MedDINOv3Wrapper`**: khung sườn kết nối backbone MedDINOv3 (sẽ nạp trọng số ở giai đoạn 2).
- **`FastVLMWrapper`**: chuẩn bị sẵn để so sánh/hoặc khai thác đặc trưng (tuỳ chiến lược thí nghiệm).
- **`MultiModalEncoder`**: chỗ đứng để bổ sung đầu vào văn bản/nhãn mô tả nếu mở rộng VLM sau này.
- **ViT (timm)**: lựa chọn baseline nhanh (vit/swin…) để kiểm tra đường dữ liệu/huấn luyện trước khi chuyển sang MedDINOv3.

> Trạng thái hiện tại: **chưa có vòng lặp train**, mới **test forward** để xác nhận I/O & tensor shapes.

---

## IV. Kiểm thử & Minh hoạ

- **Sanity check:** chạy 1 batch qua `DataLoader` → `model.forward` thành công, không lỗi shape.
- **Visualization (tuỳ chọn):** cell vẽ minh hoạ ảnh sau preprocessing (đang để optional, có thể bật khi cần báo cáo).

---

## V. Vấn đề gặp phải & Cách khắc phục

1. **Xung đột thư viện pip (trên Colab)**
   - Thông báo: `umap-learn 0.5.9.post2 requires scikit-learn>=1.6, but you have scikit-learn 1.5.2`.
   - **Giải pháp khuyến nghị:**
     - Nếu **không dùng** `umap-learn` trong Stage 1 ⇒ gỡ hoặc bỏ qua cảnh báo.
     - Nếu **cần** UMAP cho trực quan hoá sau này ⇒ nâng `scikit-learn` lên `>=1.6` **sau** khi đã cài đúng `torch`/`timm` để tránh xung đột (test lại tính tương thích CUDA).
2. **Cell subset CSV/print bị đứt mạch f-string**
   - Dòng in `print(f"Using full CSV with {len(df...}")` bị cắt giữa chừng → dễ gây lỗi runtime.
   - **Khắc phục:** hoàn thiện câu lệnh in & đảm bảo `df_all` được gán trước khi dùng; kiểm tra biến `CSV_PATH` tồn tại.
3. **Độ trễ lần truy cập đầu với GCS**
   - **Khắc phục:** bật cache tên file→folder; có thể tạo CSV phụ “filename→folder” sau 1 lần quét để lần sau load nhanh.

---

## VI. Kế hoạch tuần tới (Stage 2: Baseline huấn luyện)

1. **Gắn nhãn đa nhãn (multi‑label)** từ `Data_Entry_2017.csv` → vector 14 bệnh cho mỗi ảnh.
2. **Tích hợp MedDINOv3** (nạp trọng số):
   - Freeze backbone (giai đoạn 2a), thêm **linear head** đa nhãn (14 outputs), loss **BCEWithLogits**.
   - Thiết lập **class weights/focal loss** để xử lý mất cân bằng.
3. **Huấn luyện baseline** trên subset mở rộng (ví dụ 10–20k ảnh):
   - Mixed precision (AMP), checkpoint, early stopping.
   - **Metrics:** AUROC (micro/macro & từng lớp), F1 (threshold‑tuning), mAP.
4. **Logging & tái lập:**
   - Thiết lập TensorBoard/W&B, log seed, cấu hình, thời gian I/O.
5. **Báo cáo giữa kỳ (tuần sau):** bảng kết quả baseline + vài heatmap/Grad‑CAM minh hoạ (nếu kịp).

---

## VII. Rủi ro & Giảm thiểu

- **Băng thông GCS/Colab không ổn định:** chuẩn bị phương án cache cục bộ theo batch nhỏ nếu cần.
- **Giới hạn thời gian Colab:** chia nhỏ thí nghiệm; tự động resume từ checkpoint.
- **Mất cân bằng nhãn:** dùng class weights/focal loss, sampling có trọng số.

---

## VIII. Đề nghị xin ý kiến GV

1. **Xác nhận** lựa chọn đọc ảnh qua GCS thay vì tải về Drive (tiết kiệm dung lượng & thời gian).
2. **Metric chính** cho báo cáo: AUROC micro/macro + F1 từng lớp có phù hợp không?
3. Có cần **thêm nhiệm vụ phụ** (segmentation/Grad‑CAM) ngay ở giai đoạn 2 hay để giai đoạn 3?

---

## IX. Phụ lục

### A) Sơ đồ luồng dữ liệu (tóm tắt)

**GCS (ảnh)** → `gcsfs` → `GCSNIHCXRDataset` → `CXRPreprocessor` (+Augment) → `DataLoader` → `ModelWrapper (MedDINOv3/ViT)` → (Metrics/Logging)

### B) Cấu trúc thư mục (dự kiến)

```
/NIH_CXR
 ├─ Data_Entry_2017.csv           # CSV nhãn (trên Drive)
 ├─ cache/filename_to_folder.csv  # (tạo sau lần quét đầu tiên)
 └─ notebooks/NIH_CXR_Pipeline_Stage1_GCSstream.ipynb
```

### C) Tham số khởi tạo (hiện tại)

- `IMAGE_SIZE = 512`, `BATCH_SIZE = 8`, `NUM_WORKERS = 2`, `subset_frac = 0.02`.

> Báo cáo này phản ánh đúng trạng thái notebook **Stage 1**: đã hoàn thiện luồng IO & tiền xử lý, sẵn sàng bước sang **Stage 2: baseline training**.

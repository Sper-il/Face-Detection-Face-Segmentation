# Implementation Plan: [Tên feature/model]

## A. Research Phase

> Tham khảo model/kiến trúc liên quan (VD: UNet, Transformer...)

1. **Mục tiêu (Model Usage)**
   [Model dùng để làm gì, giải quyết bài toán gì]

2. **Model Architecture**
   [Kiến trúc được chọn, lý do chọn]

3. **Hyperparameters**
   [VD: learning rate, số layer, freeze/unfreeze layer nào, cấu hình Transformer...]

4. **Input Implementation**
   [Cách xử lý / chuẩn hóa input trước khi đưa vào model]

5. **Post-process Input**
   [Xử lý hậu kỳ output sau khi model chạy xong]

## B. Build Flow

1. **Build**
   [Triển khai code theo research phase ở trên]

2. **Test for Flow (Unit Test)**
   [Unit test cho từng bước trong flow, đảm bảo từng phần chạy đúng]

3. **Evaluation**
   - **Model evaluation:** Accuracy / Recall / Response time
   - **Full pipeline evaluation:** Test end-to-end toàn bộ flow

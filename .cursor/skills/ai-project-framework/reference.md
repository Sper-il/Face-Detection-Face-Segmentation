# Framework Phát Triển Dự Án AI

*(Tổng hợp & hệ thống hóa từ bộ khung quy trình làm việc với AI)*

## Mục lục
1. [Quy trình tổng thể (End-to-end Flow)](#1-quy-trình-tổng-thể-end-to-end-flow)
2. [Bộ câu hỏi bắt buộc trước khi vào "How" (5W1H)](#2-bộ-câu-hỏi-bắt-buộc-5w1h)
3. [Bộ file quản lý dự án](#3-bộ-file-quản-lý-dự-án)
4. [Working Rules — Quy tắc làm việc](#4-working-rules--quy-tắc-làm-việc-chung)
5. [Domain Expertise — Kiến trúc AI Agent 3 lớp](#5-domain-expertise--kiến-trúc-ai-agent-3-lớp)
6. [Evaluation Framework](#6-evaluation-framework--đánh-giá-2-cấp-độ)
7. [Context — Yếu tố quan trọng khi làm việc với AI](#7-context--yếu-tố-quan-trọng-khi-làm-việc-với-ai)
8. [Audit & Metric Framework](#8-audit--metric-framework-audit-codebase--log)
9. [Template: Progress Status](#9-template-progress-status)
10. [Template: Description](#10-template-description)
11. [Template: Implementation Plan](#11-template-implementation-plan)
12. [Template: Working Rules](#12-template-working-rules)

> 📝 **Ghi chú:** một vài chỗ trong bản ghi chú gốc khá tắt (VD "acc, t, response", "fresman"...). Mình đã suy luận theo nghĩa hợp lý nhất trong ngữ cảnh AI/ML (accuracy - time - response quality; freeze/unfreeze layer...). Bạn rà lại phần này và chỉnh nếu mình hiểu chưa đúng ý.

---

## 1. Quy trình tổng thể (End-to-end Flow)

```
Business Understanding (User's Requirements)
        │
        ▼
     Features
        │
        ▼
   Tech Solution
        │
        ▼
  Logic (AI Solution)
        │
        ▼
Implementation (Plan → Vibe Coding)
        │
        ▼
       Demo
        │
        ▼
       Test
   ├── Model level (Accuracy, Time, Response)
   └── Full pipeline (end-to-end)
```

| # | Giai đoạn | Câu hỏi cốt lõi | Output cần có |
|---|---|---|---|
| 1 | Business Understanding | Người dùng thực sự cần gì? | Requirement doc |
| 2 | Features | Yêu cầu đó = những feature cụ thể nào? | Feature list (in-scope / out-of-scope) |
| 3 | Tech Solution | Dùng công nghệ/kiến trúc gì để làm? | Tech stack + lý do chọn |
| 4 | Logic (AI Solution) | Phần AI xử lý theo logic nào? | AI solution design (model/prompt/agent...) |
| 5 | Implementation | Triển khai theo plan nào rồi mới code | Implementation plan + code |
| 6 | Demo | Kết quả chạy thử trông ra sao? | Demo build |
| 7 | Test | Có đúng và có chạy ổn không? | Test report (model level + full pipeline) |

**Nguyên tắc quan trọng:** "Vibe coding" (code nhanh theo mạch cảm hứng, có AI hỗ trợ) chỉ nên diễn ra ở bước Implementation, và **chỉ sau khi đã có plan rõ ràng**. Không nhảy thẳng từ Business Understanding sang code.

---

## 2. Bộ câu hỏi bắt buộc (5W1H)

Trước khi triển khai bất kỳ feature/task nào, phải trả lời đủ:

| Câu hỏi | Ý nghĩa | Khi nào trả lời |
|---|---|---|
| **What?** | Đang làm cái gì, phạm vi tới đâu | Đầu tiên |
| **Why?** | Tại sao chọn hướng/giải pháp này (so với phương án khác) | Ngay sau What |
| **Where / When / Which** | Áp dụng ở đâu, thời điểm nào, chọn phương án nào | Trước khi triển khai |
| **How** | Cách làm cụ thể, kỹ thuật, công cụ | **Cuối cùng** |

**Nguyên tắc:** *How* là kết quả của quá trình trả lời 3 câu hỏi trên, không phải điểm bắt đầu. Nếu ai đó (người hoặc AI agent) đề xuất thẳng "How" mà chưa có What/Why/Where-When-Which rõ ràng → dừng lại, yêu cầu làm rõ trước.

---

## 3. Bộ file quản lý dự án

Mỗi dự án/feature nên có tối thiểu các phần sau (đã đóng gói sẵn thành template ở mục 9–12 bên dưới):

| Template | Mục đích |
|---|---|
| Progress Status | Theo dõi tiến độ: fullname, description, goal, pipeline, subtasks |
| Description | Mô tả chi tiết requirement, purpose, input, output của 1 feature |
| Implementation Plan | Kế hoạch triển khai kỹ thuật: research → build → test → evaluation |
| Working Rules | Naming/folder convention, discussion & edition rule |

**Cách dùng:** copy phần template tương ứng thành file riêng cho từng feature/module, đổi tên theo feature (VD: `progress_status_ocr_module.md`), rồi điền nội dung.

---

## 4. Working Rules — Quy tắc làm việc chung

Đây là các rule mà **cả người và AI agent hỗ trợ code** đều phải tuân theo (dùng làm "AI reference" — context bắt buộc AI phải đọc trước khi code):

- **Naming Convention** — quy tắc đặt tên file/biến/hàm/branch
- **Folder Convention** — cấu trúc thư mục chuẩn của repo
- **Discussion Rule** — quy tắc khi cần thảo luận/quyết định kỹ thuật (thảo luận ở đâu, ghi quyết định thế nào)
- **Edition Rule** — quy tắc khi sửa code/tài liệu đã có (review trước khi sửa? version hóa ra sao?)

Chi tiết & mẫu điền: xem mục 12.

---

## 5. Domain Expertise — Kiến trúc AI Agent (3 lớp)

```
                         AI Agent
                            │
              ┌─────────────┴─────────────┐
              │                           │
         Prompting                       Rule
              │                           │
    Description ──▶ Skill          (Make decision)
              │                           │
     [Behavioral Layer]           [Decision Layer]
              │                           │
              └─────────────┬─────────────┘
                             ▼
                       Brain Layer
                   (The Second Brain)
                     Knowledge Base
```

| Lớp | Thành phần | Vai trò |
|---|---|---|
| **Behavioral Layer** | Description → Skill | Mô tả (description) được biên soạn thành Skill cụ thể → quyết định agent *làm gì / hành xử ra sao* |
| **Decision Layer** | Rule | Tập rule để agent *ra quyết định* — khi nào dùng skill nào, khi nào hỏi lại, khi nào từ chối |
| **Brain Layer (Second Brain)** | Knowledge Base | Kho tri thức nền mà cả 2 lớp trên tham chiếu để hoạt động chính xác, nhất quán |

**Nguyên tắc thiết kế agent:** Skill (hành vi) và Rule (quyết định) đều phải "neo" vào Knowledge Base — nếu Brain layer thiếu/sai, cả Skill lẫn Rule đều sẽ hoạt động sai dù logic viết đúng.

---

## 6. Evaluation Framework — Đánh giá 2 cấp độ

| Cấp độ | Đo lường | Khi nào dùng |
|---|---|---|
| **Model level** | Accuracy, Precision/Recall, Response time, chất lượng output riêng của model | Đánh giá model/AI logic độc lập, trước khi ráp vào hệ thống |
| **Full pipeline** | End-to-end: input thật → output thực tế, latency tổng, tỷ lệ lỗi hệ thống | Đánh giá sau khi đã tích hợp toàn bộ flow |

⚠️ **Không được dừng ở model level rồi kết luận cả hệ thống ổn** — model tốt không đảm bảo pipeline tốt (lỗi có thể nằm ở data pipeline, integration, latency cộng dồn...).

---

## 7. CONTEXT — Yếu tố quan trọng khi làm việc với AI

> Context là yếu tố quyết định khi làm việc với AI: thiếu context đúng & đủ → AI solution sai hướng dù kỹ thuật đúng.

**Khái niệm liên quan:**

- **AI Fusion** — kết hợp nhiều model/kỹ thuật/nguồn AI khác nhau để giải bài toán, thay vì phụ thuộc 1 cách tiếp cận duy nhất.
- **Enterprise Brain (The Second Brain)** — kho tri thức tổ chức tập trung, lưu context/domain knowledge/quy trình/quyết định để AI agent tham chiếu xuyên suốt dự án.

**Các vai trò (roles) cần có trong team AI:**

| Vai trò | Trách nhiệm chính |
|---|---|
| Product Owner | Định nghĩa yêu cầu, độ ưu tiên, giá trị sản phẩm |
| Domain Expert | Kiến thức chuyên môn ngành, đảm bảo đúng nghiệp vụ |
| AI Architecture | Thiết kế kiến trúc hệ thống AI tổng thể |
| AI Engineer | Xây dựng, train, deploy model/logic AI |
| AI Operator | Vận hành & giám sát hệ thống AI sau deploy |
| Security | Bảo mật dữ liệu & hệ thống AI |

---

## 8. Audit & Metric Framework (Audit Codebase / Log)

Dùng khi cần audit lại 1 dự án AI (của mình hoặc review dự án khác):

```
Metric
├── Problem Define         → Vấn đề đã được định nghĩa rõ chưa?
├── Feature → Out of Scope → Feature nào có, feature nào chủ động KHÔNG làm (ghi rõ lý do)
├── Solution → Tech → AI   → Giải pháp chọn công nghệ gì, phần AI cụ thể là gì
├── Implementation         → Đã code đến đâu, chất lượng ra sao
└── Evaluation
      ├── Model level
      └── Full pipeline
```

Audit checklist này giúp phát hiện nhanh mảnh nào đang thiếu trong 1 dự án AI (VD: có Implementation nhưng chưa có Evaluation full pipeline → dự án chưa sẵn sàng production).

---

## 9. Template: Progress Status

```markdown
# Progress Status: [Fullname - Tên đầy đủ feature/task]

**Ngày tạo:** [dd/mm/yyyy]
**Người phụ trách:** [tên]
**Trạng thái tổng:** ⬜ Not started / 🟡 In progress / 🟢 Done

---

## 1. Description
[Mô tả ngắn gọn feature/task này làm gì]

## 2. Goal / Purpose
[Mục tiêu, mục đích — giải quyết vấn đề gì, cho ai]

## 3. Pipeline
[Mô tả / sơ đồ luồng xử lý của feature này]

Input → Bước 1 → Bước 2 → ... → Output

## 4. How to do / Execution Plan (Subtasks)

| # | Subtask | Người phụ trách | Trạng thái |
|---|---|---|---|
| 1 | | | ⬜ |
| 2 | | | ⬜ |
| 3 | | | ⬜ |

## 5. Test Checklist
- [ ] Model level test đạt yêu cầu
- [ ] Full pipeline test đạt yêu cầu
- [ ] Demo sẵn sàng

## 6. Milestone Log

| Ngày | Cập nhật |
|---|---|
| | |
```

---

## 10. Template: Description

```markdown
# Description: [Tên feature/module]

## 1. Show Requirement
[Trích lại / diễn giải yêu cầu gốc từ Business Understanding]

## 2. Purpose

### 2.1 Flow
[Luồng xử lý tổng quát — liệt kê các bước tuần tự]

### 2.2 Survey
- **Capability hiện có:** [nội bộ đang có gì, làm được đến đâu]
- **Competitor / Benchmark:** [so sánh với giải pháp/sản phẩm khác trên thị trường]

### 2.3 Pipeline
[Pipeline kỹ thuật chi tiết — mô tả hoặc sơ đồ]

## 3. Input

| Thuộc tính | Chi tiết |
|---|---|
| Data source | |
| Data description | |
| Data split (train / validation / test) | |
| Định dạng ảnh xuất (nếu có) | .png |

## 4. Output
- **Raw result:** Xuất kết quả chạy ra file dữ liệu riêng (VD: .csv, .json)
- **Doc update:** Đẩy phần phân tích/tổng hợp kết quả vào file .md tương ứng (VD: results_[feature].md)
```

---

## 11. Template: Implementation Plan

```markdown
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
```

---

## 12. Template: Working Rules

```markdown
# Working Rules (AI Reference)

> File này là "context bắt buộc" — AI agent/dev đọc trước khi bắt đầu code trên dự án.

## 1. Naming Convention
- File: [VD: snake_case.py, kebab-case.md]
- Biến/hàm: [VD: snake_case cho Python, camelCase cho JS]
- Class: [VD: PascalCase]
- Branch git: [VD: feature/ten-feature, fix/ten-loi]

## 2. Folder Convention
project/
├── data/
├── src/
├── docs/
├── models/
├── tests/
└── progress_status/

[Điều chỉnh theo cấu trúc thực tế của dự án]

## 3. Discussion Rule
- Nơi thảo luận: [VD: Slack channel, GitHub Discussion, comment trong PR]
- Khi có bất đồng kỹ thuật: [quy trình quyết định, ai chốt cuối cùng]
- Ghi lại quyết định ở: [VD: ADR (Architecture Decision Record) trong docs/decisions/]

## 4. Edition Rule
- Sửa code: [có cần PR + review không, ai được merge trực tiếp]
- Sửa tài liệu (description, implementation plan): [ai được sửa, có cần thông báo team không]
- Versioning: [cách đánh version, changelog ở đâu]
```

---

## Tóm tắt nhanh (Quick Reference)

- ✅ Flow: Business → Feature → Tech → AI Logic → Implementation → Demo → Test
- ✅ Luôn trả lời What / Why / Where-When-Which trước khi trả lời How
- ✅ Mỗi feature có 3 file: progress_status + description + implementation plan
- ✅ AI Agent = Skill (behavior) + Rule (decision), cả 2 đều neo vào Knowledge Base
- ✅ Evaluate ở cả model level lẫn full pipeline — không bỏ qua bước nào
- ✅ Context là tố quan trọng — dùng Enterprise Brain để AI luôn có đủ context làm việc

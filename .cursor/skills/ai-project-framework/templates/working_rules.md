# Working Rules (AI Reference)

> File này là "context bắt buộc" — AI agent/dev đọc trước khi bắt đầu code trên dự án.

## 1. Naming Convention
- File: [VD: snake_case.py, kebab-case.md]
- Biến/hàm: [VD: snake_case cho Python, camelCase cho JS]
- Class: [VD: PascalCase]
- Branch git: [VD: feature/ten-feature, fix/ten-loi]

## 2. Folder Convention
```
project/
├── data/
├── src/
├── docs/
├── models/
├── tests/
└── progress_status/
```

[Điều chỉnh theo cấu trúc thực tế của dự án]

## 3. Discussion Rule
- Nơi thảo luận: [VD: Slack channel, GitHub Discussion, comment trong PR]
- Khi có bất đồng kỹ thuật: [quy trình quyết định, ai chốt cuối cùng]
- Ghi lại quyết định ở: [VD: ADR (Architecture Decision Record) trong docs/decisions/]

## 4. Edition Rule
- Sửa code: [có cần PR + review không, ai được merge trực tiếp]
- Sửa tài liệu (description, implementation plan): [ai được sửa, có cần thông báo team không]
- Versioning: [cách đánh version, changelog ở đâu]

# 论文目录

请将刘春生教授团队发表的论文 PDF 文件放入此目录。

## 使用步骤

1. 将 PDF 文件复制到此目录
2. 运行索引构建命令：
   ```bash
   python build_index.py
   ```
3. 启动/重启服务即可生效

## 支持的格式

- `.pdf` / `.PDF`
- `.xlsx` / `.XLSX`（每行一篇论文摘要）
- `.docx` / `.DOCX`
- `*_ocr.txt`（PDF 是扫描版时，先 OCR 成同名 `_ocr.txt` 放在此目录）

## 注意事项

- 支持中英文 PDF
- 扫描版 PDF（纯图片）可能无法提取文本，建议使用文字版
- 新增论文后需要重新运行 `build_index.py`
- 索引数据存储在 `chroma_db/` 目录，已在 `.gitignore` 中忽略

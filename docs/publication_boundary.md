# Publication Boundary

本包按 code-only 公开边界构建，只保留技术过程，不保留数据和结果。

## 本包可公开内容

- 源代码。
- 专业目录与专业细分映射。
- LLM 配置模板。
- 数据 schema。
- 技术流程说明。

## 本包刻意排除内容

- 真实用户数据库。
- 真实 profile 样本。
- 事实新闻 CSV。
- 未来新闻 JSONL。
- LLM prompt 样例。
- 原始模型返回。
- profile 级评分明细。
- 聚合结果表。
- 趋势图。
- 日志、pid、缓存、临时目录。
- API key 或 `.env`。

## 结果发布建议

如果后续需要公开结果，应单独构建 result package，并只发布经过隐私检查和人工复核的聚合表、图表和说明文档。profile 级文本、原始 LLM 输出和 prompt 样例默认不应公开。


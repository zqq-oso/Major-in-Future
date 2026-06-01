# Data Schema

本包不附带新闻数据或 profile 数据。运行者需要自行准备以下输入。

## Profile JSONL

每行一个 JSON 对象。调查脚本会使用以下字段：

| 字段 | 含义 |
| --- | --- |
| `id` | profile 内部标识。 |
| `source_id` | 来源追踪标识，可为空。 |
| `province` | 省份或地区标签。 |
| `age` | 年龄段。 |
| `gender` | 性别。 |
| `education` | 教育水平。 |
| `consumption` | 消费水平。 |
| `post_snippet` | 用户发言摘录或合成画像文本。 |

可选字段：

| 字段 | 含义 |
| --- | --- |
| `profile_text` | 如果存在，脚本会优先使用该完整文本作为画像描述。 |

## News JSONL

`--news-dir` 指向一个目录，目录内按半年期命名文件，例如：

```text
2026-S1.jsonl
2026-S2.jsonl
```

每行一个 JSON 对象：

| 字段 | 含义 |
| --- | --- |
| `date` | 日期或半年期。 |
| `domain` | 科技或产业领域。 |
| `category` | 事件类别。 |
| `title` | 新闻标题。 |
| `summary` | 新闻摘要。 |

## Historical Seed News CSV

`--seed-news-csv` 指向历史事实新闻 CSV。脚本期望中文表头：

| 字段 | 含义 |
| --- | --- |
| `时间` | 日期，例如 `2025-03-15`。 |
| `领域` | 科技或产业领域。 |
| `事件类别` | 事件类别。 |
| `新闻标题` | 新闻标题。 |
| `新闻摘要` | 新闻摘要。 |

## Run Outputs

正式运行会生成 profile 级过程输出。默认不要公开：

- `prompt_samples.jsonl`
- `raw_outputs.jsonl`
- `parsed_ratings.csv`
- `failures.csv`
- `news_contexts.json`
- `score_contexts.json`


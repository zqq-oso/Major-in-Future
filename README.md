# major_in_future_code

基于多智能体社会模拟的未来专业前景预测。

这是“未来热门学科专业预测”的 code-only 技术过程包。它只保留完整流程代码、专业目录、专业细分口径、LLM 配置模板和流程说明；不包含新闻数据、用户画像数据、实验输出、图表、日志或 API key。

## 包含内容

- `job_sim.py`：较早的端到端蒙特卡洛流程，包含新闻情境演化、智能体预测、embedding、聚类和簇命名。
- `main.py`：从本地数据源构造人口画像池的脚本框架。
- `start_full_run.py`：通过标准输入读取 API key 并后台启动完整模拟的运行辅助脚本。
- `postprocess_embedding_variants.py`：对已完成 run 进行不同 embedding 配置的后处理辅助脚本。
- `cluster_local.py`：本地 sentence-transformers embedding 辅助脚本。
- `scripts/`：正式专业前景调查、分组调查、聚合、绘图、reasoning 汇总、新闻整理和月份修正脚本。
- `major/`：99 个专业大类目录，以及专业细分群组映射文件。
- `configs/llm/llm_config.json`：OpenAI-compatible LLM/embedding 配置模板，只使用环境变量占位。
- `docs/`：输入数据 schema、技术流程和公开边界说明。

## 不包含内容

- 不包含 `data/xiaohongshu.db`。
- 不包含 `data/news/` 下的事实新闻或未来新闻。
- 不包含 `data/profile_samples/` 下的 profile 样本。
- 不包含 `output/` 下任何正式实验结果、图表、原始返回、聚合表或 prompt 样例。
- 不包含 `.env`、API key、供应商私有配置或本地日志。

## 外部输入要求

运行完整流程时，使用者需要自行提供三类输入：

1. profile JSONL：每行一个模拟个体画像，字段见 `docs/data_schema.md`。
2. news JSONL/CSV：按半年期组织的科技新闻或事实新闻种子，字段见 `docs/data_schema.md`。
3. LLM 配置：编辑 `configs/llm/llm_config.json`，并通过环境变量提供 API key。

## prepare-only 检查

在没有 API 调用的情况下，可以先检查任务组织是否正常。示例命令如下，路径需要替换为你自己的输入数据：

```bash
python scripts/run_major_outlook_survey_by_group.py \
  --profiles path/to/profiles.jsonl \
  --major-categories major/major_categories.md \
  --news-dir path/to/news_by_period \
  --seed-news-csv path/to/seed_news.csv \
  --history-end 2026-S1 \
  --output-dir output/prepare_check \
  --llm-config configs/llm/llm_config.json \
  --start-period 2026-S1 \
  --end-period 2030-S2 \
  --news-start-period 2020-S1 \
  --previous-periods 6 \
  --score-history-periods 6 \
  --news-items-per-period 10 \
  --major-ratio 1.0 \
  --limit-profiles 1000 \
  --split-door 工学 \
  --split-count 2 \
  --prepare-only
```

## 正式运行

配置环境变量：

```bash
export MODEL_API_KEY_1="your_api_key"
```

然后去掉 `--prepare-only` 运行。正式运行会在指定 `--output-dir` 下生成过程文件，包括 `metadata.json`、`news_contexts.json`、`score_contexts.json`、`prompt_samples.jsonl`、`raw_outputs.jsonl`、`parsed_ratings.csv` 和 `failures.csv`。这些文件默认不应公开发布。

## 聚合与绘图

完成一次调查后，可以运行：

```bash
python scripts/aggregate_major_outlook_trends.py --run-dir output/your_run
python scripts/plot_major_outlook_trends.py --analysis-dir output/your_run/analysis
python scripts/plot_combined_major_outlook_trends.py --analysis-dir output/your_run/analysis
python scripts/plot_door_major_outlook_trends.py --analysis-dir output/your_run/analysis
python scripts/plot_door_overall_outlook_trends.py --run-dir output/your_run
python scripts/plot_fine_group_outlook_trends.py \
  --run-dir output/your_run \
  --fine-groups major/major_category_fine_groups_v2.csv
```

聚合表和图表属于实验结果，不包含在本 code-only 包中。

## 许可证

本包没有自动附加开源许可证。正式公开前请按项目要求补充 `LICENSE`。

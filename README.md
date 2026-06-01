# Major in Future

从新闻情境到专业趋势：一个基于多智能体社会模拟的未来热门学科预测系统。

这个项目把“未来哪些学科专业会变热门”转化为一个可运行的模拟调查问题。系统不依赖单次 LLM 直接给出榜单，而是构造一组带有人口画像的模拟个体，让它们在相同的科技新闻情境下分别评价不同专业的就业前景和薪酬前景，再把个体判断聚合成专业趋势。

核心思想接近一次虚拟社会调查：每个智能体代表一个具体的社会位置和判断视角，它可能来自不同地区、年龄段、教育背景和消费层次，也可能因为自身生活经验而更关注就业稳定、薪资上限、技术替代、产业扩张或专业门槛。最终结果不是某个模型的一次性判断，而是多个模拟个体观点汇总后的群体信号。

## Method Overview

系统由四个主要层次组成。

1. **新闻情境层**  
   按半年期组织科技与前沿产业新闻，为每个调查时期提供外部环境。历史期新闻可以来自事实新闻库；未来期新闻可以由上游情境推演模块生成，也可以由研究者自行准备。未来新闻在方法上是情境变量，不被视为已经发生的事实。

2. **智能体社会模拟层**  
   每个 profile 被转化为一个画像化智能体。画像字段包括省份、年龄、性别、教育水平、消费水平和发言摘录。智能体在评分时会站在自身画像视角下作答，从而形成多元化判断。

3. **专业评分层**  
   系统要求智能体对专业大类分别给出两个 1-10 分评分：就业前景和薪酬前景。每个评分同时附带 1-2 句理由，用于后续解释专业趋势的来源。

4. **群体聚合层**  
   LLM 输出被解析为结构化表格，再按时期、专业和专业方向计算均分、排名和趋势。单个智能体的回答只是一个样本，真正进入分析的是群体评分分布及其时间变化。

## Simulation Design

推荐的正式调查入口是：

```bash
python scripts/run_major_outlook_survey_by_group.py
```

它会完成以下步骤：

1. 读取 `major/major_categories.md` 中的 99 个专业大类。
2. 按门类组织专业组，避免一次性让模型评价全部专业。
3. 为每个调查时期构造新闻上下文。
4. 为每个 `时期 × 智能体 × 专业组` 调用 LLM。
5. 解析每个专业的就业评分、薪酬评分和理由。
6. 支持断点续跑和缺失任务修复。

如果启用 `--score-history-periods`，当前期智能体还会看到前若干期同专业组的历史统计均分。这个设置相当于为模拟加入弱形式的社会记忆，使后续判断不只是对当前新闻的即时反应，也会参考此前群体评价轨迹。

## Repository Structure

```text
.
├── configs/
│   └── llm/llm_config.json              # OpenAI-compatible 模型配置模板
├── docs/
│   ├── data_schema.md                   # profile 与 news 输入格式
│   ├── publication_boundary.md          # 公开发布边界建议
│   └── technical_pipeline.md            # 技术流程说明
├── major/
│   ├── major_categories.md              # 99 个专业大类
│   ├── major_category_fine_groups.csv   # 专业细分映射
│   └── major_category_fine_groups_v2.csv
├── scripts/
│   ├── run_major_outlook_survey.py
│   ├── run_major_outlook_survey_by_group.py
│   ├── aggregate_major_outlook_trends.py
│   ├── plot_major_outlook_trends.py
│   ├── plot_combined_major_outlook_trends.py
│   ├── plot_door_major_outlook_trends.py
│   ├── plot_door_overall_outlook_trends.py
│   └── plot_fine_group_outlook_trends.py
├── job_sim.py                           # 早期端到端蒙特卡洛流程
├── main.py                              # profile 池构造脚本框架
├── cluster_local.py                     # 本地 embedding 辅助脚本
└── start_full_run.py                    # 后台运行辅助脚本
```

## Input Data

运行调查前需要准备三类输入。

### 1. Profile JSONL

每行一个模拟个体画像。核心字段包括：

```json
{
  "id": 1,
  "source_id": "synthetic-001",
  "province": "广东省",
  "age": "19-35岁",
  "gender": "女",
  "education": "本科及以上",
  "consumption": "中等",
  "post_snippet": "一段用于刻画生活场景和表达风格的文本"
}
```

更完整的字段说明见 [docs/data_schema.md](docs/data_schema.md)。

### 2. News By Period

`--news-dir` 指向一个按半年期组织的新闻目录，例如：

```text
data/news_by_period/
├── 2026-S1.jsonl
├── 2026-S2.jsonl
└── 2027-S1.jsonl
```

每行包含 `date`、`domain`、`category`、`title` 和 `summary` 字段。

### 3. Historical Seed News CSV

`--seed-news-csv` 用于补充历史事实新闻。脚本期望中文表头：

```text
时间,领域,事件类别,新闻标题,新闻摘要
```

## Model Configuration

模型配置文件位于：

```text
configs/llm/llm_config.json
```

它使用 LiteLLM Router 和 OpenAI-compatible API。配置示例：

```json
{
  "chat_models": [
    {
      "model_name": "chat",
      "litellm_params": {
        "model": "openai/your-chat-model",
        "api_base": "https://your-openai-compatible-endpoint/v1",
        "api_key": "os.environ/MODEL_API_KEY_1",
        "max_parallel_requests": 20
      }
    }
  ]
}
```

运行前设置环境变量：

```bash
export MODEL_API_KEY_1="your_api_key"
```

## Prepare-only Check

可以先运行 prepare-only 检查任务组织、专业分组和上下文构造，不会调用 LLM：

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

## Run A Survey

去掉 `--prepare-only` 后即可正式运行。建议为每次实验指定独立输出目录：

```bash
python scripts/run_major_outlook_survey_by_group.py \
  --profiles path/to/profiles.jsonl \
  --news-dir path/to/news_by_period \
  --seed-news-csv path/to/seed_news.csv \
  --output-dir output/major_outlook_run \
  --llm-config configs/llm/llm_config.json \
  --start-period 2026-S1 \
  --end-period 2030-S2 \
  --previous-periods 6 \
  --score-history-periods 6 \
  --news-items-per-period 10 \
  --limit-profiles 1000 \
  --resume
```

运行完成后，输出目录会包含 `metadata.json`、`major_groups.csv`、`news_contexts.json`、`score_contexts.json`、`raw_outputs.jsonl`、`parsed_ratings.csv` 和 `failures.csv` 等过程文件。

## Aggregate And Plot

对一次完整运行做聚合：

```bash
python scripts/aggregate_major_outlook_trends.py --run-dir output/major_outlook_run
```

生成专业趋势图：

```bash
python scripts/plot_major_outlook_trends.py \
  --analysis-dir output/major_outlook_run/analysis
```

生成组合趋势图：

```bash
python scripts/plot_combined_major_outlook_trends.py \
  --analysis-dir output/major_outlook_run/analysis
```

生成门类和细分专业方向趋势图：

```bash
python scripts/plot_door_major_outlook_trends.py \
  --analysis-dir output/major_outlook_run/analysis

python scripts/plot_door_overall_outlook_trends.py \
  --run-dir output/major_outlook_run

python scripts/plot_fine_group_outlook_trends.py \
  --run-dir output/major_outlook_run \
  --fine-groups major/major_category_fine_groups_v2.csv
```

## Method Notes

专业趋势结果应被理解为情境化社会模拟结果：它反映的是一组画像化智能体在给定新闻情境、专业目录和提示词协议下形成的群体判断。新闻、画像、模型、评分锚点和历史分数反馈都会影响最终曲线。

因此，这个系统更适合用于探索“在某种技术与产业情境下，不同专业方向可能如何被社会群体评价”，而不是直接测量现实世界未来就业市场。

## License

No license has been added yet. Add a `LICENSE` file before formal public release if needed.

# 文本-事件信号提取与事件驱动 Alpha

核心思路：把新闻、公告、财报、电话会议、监管文件、社媒等非结构化文本转成高准确度的结构化事件与因子（例如：盈利超预期/下调指引/并购传闻/高管离职），把这些信号作为事件驱动策略或因子输入到回测/下单系统。

具体用例：

- 实时识别并量化“盈利惊喜/指引变动/回购/分红/诉讼”等事件并测量其短中期价格影响。
- 从电话会议文本提取“管理层语气/确定性/关键主题转移”构造动量或反转信号。
- 对社媒/论坛做谣言/情绪过滤并结合成交量异常做短期交易信号。

技术实现要点：

- 数据管道：流式采集 → 时间戳标准化（避免未来函数）→ 清洗 → LLM + 提取器 → 结构化记录入 Feature Store。
- 模型组合：用 LLM 做高召回的文本理解/提取（NER、事件分类、因果关系提取），然后用专门的判别模型或规则（逻辑校验、置信度门槛）提升精确度。
- 工具链：RAG（retrieval-augmented generation）用于上下文丰富；向量 DB 用于相似事件检索；小型分类器/校验器做二次筛查。
- 量化化输出示例：事件类型、强度分数（-1..+1）、置信度、相关实体、时间窗口（T0, T+1…T+N）。

评价指标 & 风险：

- 指标：事件识别的 Precision/Recall、事件后短中期平均异常收益、Information Coefficient (IC)、回测的 Sharpe/最大回撤/交易成本敏感度。
- 风险：时间戳/数据泄露导致 Look-ahead；LLM 幻觉导致虚假事件；新闻重复/噪声造成信号污染。
- 缓解：严格时间对齐、引入验真模块（cross-source verification）、对事件打置信度并只在高置信时进场、在回测加上滑点与交易成本模拟。

目标产物：实时/批量把原始文本（新闻/公告/转录/社媒）转为结构化事件（event），每条事件包含：asset_id、event_type、polarity、magnitude、confidence、publish_time、evidence_text 等字段，并存入 Event Store → 被 Signalizer 转为可回测/可执行的仓位指令或警示。

关键防错点：严格时间对齐（用 source_publish_time）、防止 LLM 幻觉（schema 校验 + cross-source 验证 + 小模型二次判定）、去重（同一事件多次转载）和模拟真实交易成本（回测时）。

目标总结（一句话）：把非结构化文本（新闻、公告、财报、电话会议、社媒等）实时或批量转成结构化事件流（type, polarity, magnitude, confidence, timestamp, entities），作为量化信号输入回测与交易系统。

## 1. Pipeline

阶段：采集 → 预处理（清洗/去重/时区）→ 时间对齐（发布时戳）→ 文本理解（LLM/NER/分类/关系提取）→ 可信性验证（cross-source / rule-based）→ 事件归一化（schema）→ 写入 Feature Store / Event Store → 信号化（mapping→position sizing）→ 回测/模拟 → 交易执行（有/无人工审查）→ 监控/治理

## 2. 数据源与采集细节

- 官方：公司公告（EDGAR/港交所/上交所），监管公告，新闻稿，交易所公告。
- 报道：新闻社（路透、彭博、路透等）、行业媒体。
- 语音转文本：earnings call、管理层电话会议（用 ASR 转写并保留信心分数）。
- 社媒/论坛：Twitter/X、StockTwits、Reddit 等（注意合规与噪声）。
- 研究报告/分析师笔记（可能有版权限制）。

采集要点

- 可追溯时间戳：保存 source_publish_time（原始发布时间）和 inbound_crawl_time（我们收到时间）。回测/信号化务必用 source_publish_time。
- 去重/规范化：同一稿件的不同转载会造成重复事件——需要基于（title+body hash 或 canonical_url）去重与版本管理。
- 源信誉分：给每个来源一个权重（历史准确率、速度）以便后续 cross-source 验证。
- 语言/编码：自动检测语言并按语言路由到合适的模型/翻译器（保持原文保留以便审计）。

## 3. 时间对齐与避免未来函数

- 严格使用原始 publish_time 来决定事件发生时刻；所有 downstream 模型只可访问 <= publish_time 的市场数据。
- 延迟窗口建模：不同来源有不同延迟（如监管公告 vs 新闻社），系统需记录并在回测中模拟传播延迟（比如 0s / 1s / 30s / 300s）。
- 时区与交易日处理：统一成 UTC 并记录本地交易时；对盘前/盘中/盘后消息做不同规则（盘后消息可能第二日开盘才消化）。

## 4. 事件 schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "market_event",
  "type": "object",
  "required": ["event_id","asset_id","asset_type","source","source_publish_time","event_type","polarity","magnitude","confidence","evidence_text","raw_document_id","schema_version"],
  "properties": {
    "event_id": {"type":"string"},
    "asset_id": {"type":"string"},
    "asset_type": {"type":"string","enum":["equity","etf","bond","fx","crypto","index"]},
    "source": {"type":"string"},
    "source_reliability": {"type":"number","minimum":0,"maximum":1},
    "source_publish_time": {"type":"string","format":"date-time"},
    "ingest_time": {"type":"string","format":"date-time"},
    "event_type": {"type":"string"},
    "polarity": {"type":"number","minimum":-1,"maximum":1},
    "magnitude": {"type":"number","minimum":0,"maximum":1},
    "confidence": {"type":"number","minimum":0,"maximum":1},
    "evidence_text": {"type":"string"},
    "entities": {
      "type":"array",
      "items":{
        "type":"object",
        "required":["name","type"],
        "properties":{"name":{"type":"string"},"type":{"type":"string"},"id":{"type":"string"}}
      }
    },
    "raw_document_id": {"type":"string"},
    "related_events": {"type":"array","items":{"type":"string"}},
    "tags": {"type":"array","items":{"type":"string"}},
    "schema_version": {"type":"string"}
  }
}
```

event_type: earnings_beat, earnings_miss, guidance_up, guidance_down, merger_acquisition, lawsuit, exec_departure, analyst_upgrade, analyst_downgrade, share_buyback, dividend_change

## 5. LLM 抽取层

两类策略

1. Prompt + LLM（zero/few-shot）：快速启动，适用于高召回的初步抽取。
2. 微调/检索增强（RAG）+专门分类器：在有标注数据后，把 LLM 当作理解/解析器，再由任务特定的小模型做精确判定（提升 precision）。

常见模块

- NER（公司/人物/数值/日期/比率）
- Event classification（multi-label）
- Causal relation extraction（who did what to whom, e.g., “Company X lowered guidance”）
- Quant extraction（把“每股盈利下降 10%”映射为数值字段）
- Sentiment / Tone detection（电话会语气、CEO 表态强度）

Prompt & 输出策略（示例）

要求 LLM 只输出 JSON（严禁自由文本），并在输出前后用标记，便于解析与校验。

```json
任务：从下面的新闻文本中提取事件。输出必须为有效 JSON（UTF-8），字段必须包含：event_type, asset_ticker, polarity, magnitude（-1..1）, confidence（0..1）, evidence_text（原文片段）。如果没有可提取事件，返回 {"event_type":"none"}。

示例1输入:
"Apple 今天发布财报，每股盈利超出分析师预期 0.12 美元，公司上调全年指引。"

示例1输出:
{
 "event_type":"earnings_beat",
 "asset_ticker":"AAPL",
 "polarity":1,
 "magnitude":0.6,
 "confidence":0.95,
 "evidence_text":"每股盈利超出分析师预期 0.12 美元，公司上调全年指引"
}

现在提取:

{text}
```

Few-shot 示例：给 3–5 个典型输入/输出对以提高正确率；在 prompt 里要求 json only 并提供 schema 验证规则（例如 magnitude 取值范围）。

后处理：对 LLM 输出做 schema 校验（JSON schema），对数值做 sanity check（magnitude>1 或 < -1 则打回），并由小型二分类器对 low-confidence 输出做二次筛查。

## 6. 交叉验证与可信性 Cross-source verification

同一事件由多个来源独立报告时可信度更高

- 在时间窗口（例如 ±T 秒/分钟）内搜索其他来源类似表述；
- 计算加权共识分：consensus = Σ (source_reliability_i * indicator_i) / Σ source_reliability_i；
- 若 consensus 超阈值则升级事件优先级；否则标为“单源/低可信”并按较小仓位处理或不交易。
- 对于重大事件（并购、监管处罚等）设置人审路径。

## 7. 从事件到可交易信号（signalization）

常见映射方法

- 做方向/持仓：signal_score = polarity * magnitude * confidence * source_weight
- 大小/仓位（position sizing）：按流动性/波动率调整仓位，例如 position = k * signal_score / volatility（注意用未来不可知的波动率估计器）。
- 持有期：根据事件类型设定默认 decay（如 earnings_beat：持仓 1-5 日；merger：持仓更长）。
- 过滤：市值/最低成交量/短卖限制/风险规则过滤。

示例伪代码

```python
for event in event_stream:
    if event.confidence < 0.7: continue
    if event.event_type == "earnings_miss":
        signal = - event.magnitude * event.confidence
        size = apply_risk_limits(signal, asset)
        place_order(asset, size)
```

## 8. 回测注意

核心思路：事件在 source_publish_time 可被市场“看到”，回测在该时刻/下一可交易点创建订单，并模拟真实滑点/交易成本。

- 使用 event-time backtest：事件在 publish_time 被触发，回测在该时间点模拟下单，并考虑市场可见性（若为盘后消息则次日开盘）。
- 模拟执行成本：加入Spread、滑点模型、市场冲击；尤其是短期事件驱动策略对这些非常敏感。
- 样本可变性：对高频事件应使用 minute/bar 回测；对中长期事件可用日频。
- 避免数据泄露：把原始文本、爬取时间、修订记录都保存，确保回测环境只能读取当时可获得的文本/字段。
- 健壮性检查：做决策树回退、随机合成试验、去除单一来源的回测，以查看策略是否过度依赖个别来源或极端事件。

必须仿真/校验项
- ingestion_delay：对不同 source 设值（如监管公告 0s, 社媒 30s, 新闻社 1–3s），并在回测中模拟。
- 执行模型：Spread、market impact（按日均成交量比例模拟）以及佣金。
- 损耗/过滤：最低流动性过滤、价格跳空（盘前/盘后）处理策略。

## 9. 评估指标（技术 + 交易）

- 文本抽取指标：Precision / Recall / F1（按事件类型）、Calibration（比如 Brier score）、NER F1、数值提取误差(MAE)。
- 策略/交易指标：Event IC（事件分数与随后收益的相关）、平均异常收益、Sharpe、最大回撤、Turnover、交易成本敏感性分析、Statistical significance（t-test、bootstrap）。
- 生产监控：事件分布随时间的漂移、平均 confidence、cross-source consensus rate、触发率异常告警。

质量提升

- 人机闭环（Active Learning）：挑高不确定/低置信样本给人工标注，周期性微调/训练。
- 置信度校准：对模型输出做 Platt/Isotonic 校准，让 confidence 能代表真实概率。
- 版本控制：事件 schema、模型、数据源都版本化；保留 raw 文本用于审计。
- 安全与合规：对受版权保护数据做授权管理；对社媒/用户生成内容做额外审核。
- 性能/延迟预算：对实时策略设定处理延迟目标（例如从抓取到输出事件 < 1s / 5s / 30s 取决于策略），并对模型做并发与缓存优化。

## 10. 问题

- 幻觉（LLM“瞎编”）：严格 schema 校验 + cross-source 验证 + 低阈值下人审。
- 重复触发（same story multiple reposts）：使用 dedup + event clustering 来防止多次进场。
- 未来函数：所有回测用原始 publish_time 与爬取延迟叠加的真实可见时间。
- 过度敏感/过拟合：在回测中模拟费用与冲击；用滚动窗口 CV 测试稳定性。

Prompt 模板

```json
任务：请从下面的新闻/公告/转录中提取**结构化事件**。输出**仅为有效 JSON**（UTF-8），必须包含字段：event_type, asset_id, polarity, magnitude, confidence, evidence_text, source, source_publish_time, raw_document_id。若文本中没有任何可交易的事件，请返回 {"event_type":"none"}。

事件类别（示例）：earnings_beat, earnings_miss, guidance_up, guidance_down, merger_acquisition, executive_departure, share_buyback, dividend_change, analyst_upgrade, analyst_downgrade, legal_action, none。

输出时注意：
- polarity: +1 为利好, -1 为利空, 0 中性。
- magnitude: 0..1（估计影响强度）。
- confidence: 0..1（模型对抽取结果的信心）。
- evidence_text: 触发判断的原文摘录（简短）。

--- 示例（few-shot） ---
输入:
"苹果公司今日发布季度财报，每股盈利为1.45美元，高于市场预期的1.30美元；管理层同时上调了全年营收指引。"

输出:
{
 "event_type":"earnings_beat",
 "asset_id":"AAPL",
 "polarity":1,
 "magnitude":0.7,
 "confidence":0.95,
 "evidence_text":"每股盈利为1.45美元，高于市场预期1.30美元；上调全年营收指引",
 "source":"SampleNews",
 "source_publish_time":"2025-08-09T02:15:23Z",
 "raw_document_id":"sample-doc-001"
}

输入:
"某些社交媒体散布公司X将被收购的谣言，但官方并无公告。"

输出:
{"event_type":"none"}

--- 现在开始处理：

待处理文本
```

few-shot 示例集

1. 盈利超预期 + 上调指引（positive）
2. 盈利不及预期 + 下调指引（negative）
3. 董事长离职（可能负面，但视上下文）
4. 并购传闻（未经证实） -> 返回 none 或 low_confidence 并标 tag 表示“rumor”。
5. 回购计划宣布（positive）
6. 监管调查/罚款（negative）

preprocess, llm_wrapper, validator, postprocess, event_store, consensus, signalizer, tests

utils.py

```python
import uuid, time
from dateutil import parser

def make_event_id():
    return str(uuid.uuid4())

def parse_time(ts_str):
    return parser.isoparse(ts_str)  # returns datetime

schema_validator.py

import jsonschema, json
from jsonschema import validate

with open('event_schema.json','r') as f:
    EVENT_SCHEMA = json.load(f)

def validate_event(evt):
    try:
        validate(instance=evt, schema=EVENT_SCHEMA)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
```

event_extractor.py

```python
import json, time
from utils import make_event_id
from llm_wrapper import call_llm
from schema_validator import validate_event
from datetime import datetime

def extract_event_from_text(raw_text, source, source_publish_time, raw_document_id, prompt_template):
    prompt = prompt_template.replace("[待处理文本在此]", raw_text)
    # 在 prompt 中加入 source info 以便 LLM 输出完整字段
    prompt = prompt + f"\n\n--METADATA--\nsource:{source}\nsource_publish_time:{source_publish_time}\nraw_document_id:{raw_document_id}\n"
    raw_output = call_llm(prompt)
    # 尝试解析 JSON（若 LLM 返回多余文本需要 regex 提取）
    try:
        evt = json.loads(raw_output)
    except Exception as e:
        # 尝试从文本中抽出第一个 {...} 子串
        import re
        m = re.search(r"(\{.*\})", raw_output, re.S)
        if not m:
            return None, f"no json in llm output: {raw_output[:200]}"
        evt = json.loads(m.group(1))
    # 填补/补齐字段
    evt.setdefault("event_id", make_event_id())
    evt.setdefault("ingest_time", datetime.utcnow().isoformat()+"Z")
    evt.setdefault("source", source)
    evt.setdefault("source_publish_time", source_publish_time)
    evt.setdefault("raw_document_id", raw_document_id)
    # validate
    ok, err = validate_event(evt)
    if not ok:
        return None, err
    return evt, None
```

dedupe.py 去重/聚类同一事件

```python
from hashlib import sha1
import time

def compute_doc_hash(title, body):
    h = sha1()
    h.update((title or "").encode('utf8'))
    h.update((body or "").encode('utf8'))
    return h.hexdigest()

# 简单去重策略：对同一 asset 在 short window 内相同 hash 或高相似度视为同一事件
def is_duplicate(new_hash, recent_hashes_set):
    return new_hash in recent_hashes_set
```

consensus.py

```python
def compute_consensus(event_candidates):
    """
    输入：同一 asset 在 window 内由不同 source 提出的事件候选列表（同一 event_type）
    每个 candidate 包含 source_reliability (0..1) 与 confidence (0..1)
    输出：weighted_consensus_score (0..1) 与 combined_confidence
    """
    numerator = 0.0
    denom = 0.0
    for c in event_candidates:
        weight = c.get("source_reliability", 0.5)
        score = c.get("confidence", 0.5)
        numerator += weight * score
        denom += weight
    if denom == 0:
        return 0.0
    return numerator/denom
```

signalizer.py（事件→信号）

```python
def event_to_signal(event, asset_liquidity, volatility_estimate, risk_budget=0.01):
    """
    简单映射示例：
      - signal_score = polarity * magnitude * confidence * source_reliability
      - position_size = risk_budget * (signal_score / volatility_estimate)
    返回：{'side': 'long'/'short'/'flat', 'size': float}
    """
    sr = event.get("source_reliability", 0.8)
    score = event["polarity"] * event["magnitude"] * event["confidence"] * sr
    if abs(score) < 0.05:
        return {"side":"flat","size":0.0, "raw_score":score}
    # 简单归一化：risk_budget代表账户风险比例
    size = risk_budget * score / max(volatility_estimate, 0.0001)
    side = "long" if score > 0 else "short"
    return {"side": side, "size": abs(size), "raw_score": score}
```

event_store SQL 表

```sql
CREATE TABLE market_events (
  event_id TEXT PRIMARY KEY,
  asset_id TEXT,
  asset_type TEXT,
  source TEXT,
  source_reliability NUMERIC,
  source_publish_time TIMESTAMP WITH TIME ZONE,
  ingest_time TIMESTAMP WITH TIME ZONE,
  event_type TEXT,
  polarity NUMERIC,
  magnitude NUMERIC,
  confidence NUMERIC,
  evidence_text TEXT,
  raw_document_id TEXT,
  related_events TEXT[],
  tags TEXT[],
  schema_version TEXT
);
```

## 数据质量、校验与防错

- 严格 schema 验证（jsonschema）；任何不合格事件进入“quarantine”并记录错误原因供人工审查。
- 数值 sanity check：magnitude、confidence 范围检查；event_type 必须在枚举表内；source_publish_time 不可晚于 ingest_time（否则记录为时间错乱）。
- 数字提取的 regex 校验：LLM 抽取到数字后用正则/解析复核（例如 $1.45 → 1.45）；对于 EPS/百分比等做单位检查。
- 重复检测：对 title+body 或 evidence_text 做 Hash 去重；对相似度高（cosine>0.9）短时间内重复记录合并为同一事件。
- cross-source 验证：在 ±T 秒/分钟窗口内查找其他来源相同或相近事件。若单源且源可靠度低，则置信度下降或直接 require human_review。

## 减少 LLM 幻觉 & 提升精确度的架构建议

1. LLM 做高召回抽取 → 输给小型专用判别器（XGBoost / LightGBM / logistic）做二次过滤以提高 precision。
2. RAG（检索增强生成）：在 prompt 前将该公司的历史公告或相似事件检索入 prompt，避免 LLM “凭空推理”。
3. 正则/规则做核查：对 key 数值（EPS、指引变动）做 deterministic 解析与校验。
4. 置信度校准：对 LLM 输出的 confidence 做 Platt/Isotonic 校准（以人工标注集为基准）。
5. 人机闭环：对 low_confidence 或高影响重大事件（收购、监管处罚）强制人审。

## Active Learning 标注循环

目标：用少量人工标注提升模型精度。策略：

- 不确定性采样：挑 confidence 靠近 0.5 的样本优先标注。
- 多样性采样：用 embedding 聚类，从每个 cluster 抽样，防止只挑重复样本。
- 影响度采样：挑高 magnitude 或高市值受影响公司的样本优先标注。
- 训练迭代：人工标注 → 用标注集微调判别器 / calibration → 部署→ 再采样。

标注 schema（表格）：

text | asset_id | event_type (enum) | polarity (+/-/0) | magnitude_estimate | evidence_text | confidence_label (human) | notes
 --- | --- | --- | --- | --- | --- | --- | ---     
 
 ## 监控指标与告警（KPI）

- 输入层：抓取延迟（crawl_time - source_publish_time）平均与 P95；源覆盖率（每分钟抓取成功率）。
- 模型层：事件/分钟、平均 confidence、各 event_type 的比例、cross-source consensus rate、单源事件占比。
- 策略层：事件 IC（rolling 30/90/180 天）、事件触发的 P&L（按事件类型汇总）、回测 vs 实盘差异。

告警样例：

- 平均 confidence 突降 > 25% 相较基线 → 发告警。
- 单源事件比例 > 60% （可能数据源故障或被操纵） → 发告警。
- 抽取 latency > 5s（超出 SLA） → 运维告警。

- 幻觉：校验 JSON schema、对数值用 regex 二次解析、跨源验证、低置信时人审。
- 重复进场：基于 doc_hash + event clustering 合并重复报道。
- 时间错配：一律以 source_publish_time 做可见时间。回测时加入真实传播延迟模拟（不同来源不同延迟）。
- 过拟合回测：在回测中加入 realistic transaction cost / slippage / market impact，并通过滚动窗口 CV 测试稳定性。
- 法律/合规：社媒信号与内部交易规则审查（避免 insider info 问题）。
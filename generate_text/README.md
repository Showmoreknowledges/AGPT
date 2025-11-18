# generate_text 模块

本模块用于将多层（当前专注于两层）网络数据中的节点结构特征、跨层锚点信息以及跨层跳数/路径拼接成自然语言描述，便于将节点表示为文本输入到 TANS 的下游流程。

## 目录结构

- `generate_multilayer_text.py`：主脚本，读取配置并生成节点文本与元信息。
- `config_example.yaml`：示例配置文件，说明所需的输入数据格式。

## 依赖

与 `TANS/preprocess` 下脚本类似，需要：

- Python ≥ 3.9
- `numpy`, `pandas`, `pyyaml`, `networkx`, `torch`（仅当输入是 `.pt/.pth`）

## 输入数据说明

通过 YAML 配置提供以下内容：

1. **layers**：每一层包含
   - `name`：层名称
   - `prefix`：节点在文本中的前缀（可选）
   - `edge_path`：边列表或 edge_index；支持 `.npy/.npz/.pt/.csv/.tsv`
   - `edge_key`：当 `npz`/`pt` 中包含多个 tensor 时指定键名（可选）
   - `feature_path`：节点特征矩阵，用于确定节点数量
   - `feature_key`：同上
2. **alignment**：锚点/对齐信息
   - `path`：`csv/tsv/json`，包含两层节点的索引
   - `columns`：指明两层节点 id 所在列名
3. **hyper_adjacency**：超邻接（跨层跳数）矩阵
   - `path` 与可选 `key`
   - `axis_order`：矩阵行/列对应的层顺序，例如 `[layer_a, layer_b]`
4. **output**：文本与元数据写入路径
5. **text_options**：文本模版配置，例如要展示的结构特征、选取多少跨层目标节点、是否输出路径等。

详见 `config_example.yaml`。

## 使用方法

```bash
cd TANS/generate_text
python generate_multilayer_text.py --config config_example.yaml
```

运行后会得到：

- `text_path`：字典 `{layer_name: {node_id: "text"}}`
- `metadata_path`：包含每个节点的原始指标、锚点与跨层统计，便于调试或后续处理。

## 结果说明

每个节点的文本均包含：

1. **结构特征**：如度、聚类系数、介数等以及其在该层内的百分位信息。
2. **跨层锚点路径**：节点到最近锚点的跳数以及锚点对应另一层的节点。
3. **跨层跳数与路径**：从该节点，通过锚点抵达另一层若干代表节点的路径与总跳数；同时给出跨层距离统计摘要。

可以按需调节配置中的模板与指标，以适配不同多层网络数据。*** End Patch
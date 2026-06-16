# 日志实体提取模块

## 功能说明

本模块用于从IoTDB日志中提取以下实体信息：
- **DataRegion**: 数据区域ID（如 DataRegion[1], DataRegion[2]）
- **Node**: 节点ID和IP地址（如 nodeId=1, IP=172.20.0.11）
- **Thread**: 线程名（如 pool-34-IoTDB-LogDispatcher-DataRegion[2]-1）
- **ConsensusGroup**: 共识组信息（如 TConsensusGroupId(type:DataRegion, id:1)）

## 使用方法

### 基本用法

```python
from entity_extractor import extract_entities_from_log_line

# 从完整日志行中提取实体
log_line = "- 2024-01-12 03:09:34,967 [pool-31-IoTDB-LogDispatcher-DataRegion[1]-1] INFO ..."
entities = extract_entities_from_log_line(log_line)

# 访问提取到的实体
print(entities.data_regions)  # {'1'}
print(entities.nodes)  # {'1', '2'}
print(entities.node_ips)  # {'1': '172.20.0.11', '2': '172.20.0.12'}
print(entities.thread)  # 'pool-31-IoTDB-LogDispatcher-DataRegion[1]-1'
print(entities.consensus_groups)  # {'DataRegion:1'}

# 转换为字典格式
print(entities.to_dict())
```

### 从日志消息中提取

```python
from entity_extractor import extract_entities_from_message

# 如果只有日志消息（不含时间戳和线程）
message = "DataRegion[1]: startIndex: 1, maxIndex: 1"
thread = "pool-31-IoTDB-LogDispatcher-DataRegion[1]-1"  # 可选
entities = extract_entities_from_message(message, thread)
```

## 提取规则

### DataRegion
- **模式**: `DataRegion[数字]` 或 `root.sg数字[数字]`
- **示例**: `DataRegion[1]`, `DataRegion[2]`, `root.sg1[1]`
- **提取结果**: 数字ID（字符串格式）

### Node
- **模式1**: `nodeId=数字` 
- **模式2**: `Peer{..., nodeId=数字, endpoint=TEndPoint(ip:IP, port:端口)}`
- **示例**: `nodeId=1`, `nodeId=2`
- **提取结果**: 
  - Node ID集合
  - Node ID到IP地址的映射字典

### Thread
- **模式**: 日志行格式 `[线程名]`
- **示例**: `[pool-31-IoTDB-LogDispatcher-DataRegion[1]-1]`
- **提取结果**: 完整的线程名字符串
- **注意**: 线程名本身可能包含`[]`，已正确处理

### ConsensusGroup
- **模式**: `TConsensusGroupId(type:类型, id:数字)`
- **示例**: `TConsensusGroupId(type:DataRegion, id:1)`
- **提取结果**: 格式为 `类型:ID`（如 `DataRegion:1`）

## 数据结构

### LogEntities类

```python
@dataclass
class LogEntities:
    data_regions: Set[str]  # DataRegion ID集合
    nodes: Set[str]  # Node ID集合
    node_ips: Dict[str, str]  # Node ID -> IP地址映射
    thread: Optional[str]  # 线程名
    consensus_groups: Set[str]  # ConsensusGroup集合（格式: "type:id"）
```

## 测试

运行测试脚本：

```bash
# 运行单元测试
python3 entity_extractor.py

# 使用真实日志数据测试
python3 test_extractor.py
```

## 示例输出

```python
{
    'data_regions': ['1', '2'],
    'nodes': ['1', '2', '3'],
    'node_ips': {
        '1': '172.20.0.11',
        '2': '172.20.0.12',
        '3': '172.20.0.13'
    },
    'thread': 'pool-34-IoTDB-LogDispatcher-DataRegion[2]-1',
    'consensus_groups': ['DataRegion:1', 'DataRegion:2']
}
```

## 注意事项

1. **线程名提取**: 线程名可能包含内部的`[]`，正则表达式已处理这种情况
2. **IP地址关联**: Node的IP地址只有在日志中同时出现`nodeId`和`TEndPoint`时才会被提取
3. **合并实体**: 使用`merge()`方法可以合并多个LogEntities的结果

## 性能

- 单行日志提取时间: < 1ms
- 支持处理大量日志行
- 使用正则表达式缓存，提高效率



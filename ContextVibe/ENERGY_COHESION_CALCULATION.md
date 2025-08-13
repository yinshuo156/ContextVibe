# E、C计算方法详解

## 📋 概述

ContextVibe项目中的E（Energy，交互同步性）和C（Cohesion，上下文一致性）是两个重要的情感分析指标，用于评估会议参与者的交互状态和团队融入度。

## 🎯 E - Energy（交互同步性）

### 定义
Energy指标衡量参与者在会议中的交互活跃度和同步性，反映了个人的参与程度和与环境的协调性。

### 计算方法

#### 1. 音频Energy计算

**公式：**
```
E_audio = (Volume_Intensity + Speech_Rhythm) / 2
```

**详细步骤：**

1. **音量强度计算**
   ```python
   # 使用RMS（Root Mean Square）计算音量强度
   rms = librosa.feature.rms(y=y)
   volume_intensity = np.mean(rms)
   ```

2. **语速节奏计算**
   ```python
   # 使用节拍跟踪计算语速
   tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
   speech_rhythm = tempo / 200.0  # 归一化到[0,1]
   ```

3. **最终Energy分数**
   ```python
   energy = (volume_intensity + speech_rhythm) / 2
   energy = np.clip(energy, 0, 1)  # 限制在[0,1]范围
   ```

**数学公式：**
```
E_audio = \frac{1}{2} \left( \frac{1}{N} \sum_{i=1}^{N} RMS_i + \frac{Tempo}{200} \right)
```

其中：
- `RMS_i` 是第i帧的均方根值
- `Tempo` 是检测到的语速（BPM）
- `N` 是音频帧数

#### 2. 视频Energy计算

**公式：**
```
E_video = \frac{1}{M} \sum_{j=1}^{M} MotionScore_j
```

**详细步骤：**

1. **面部检测**
   ```python
   # 使用MediaPipe进行面部检测
   results = mp_face_mesh.process(rgb_frame)
   ```

2. **面部动作计算**
   ```python
   # 提取面部关键点
   points = np.array([[landmark.x, landmark.y, landmark.z] 
                      for landmark in landmarks.landmark])
   
   # 计算Z轴变化（面部动作）
   motion_score = np.std(points[:, 2])  # Z轴标准差
   motion_score = np.clip(motion_score * 10, 0, 1)
   ```

3. **最终Energy分数**
   ```python
   # 对所有帧的动作分数求平均
   energy = np.mean(motion_scores)
   ```

**数学公式：**
```
E_video = \frac{1}{M} \sum_{j=1}^{M} \min(10 \cdot \sigma(Z_j), 1)
```

其中：
- `Z_j` 是第j帧面部关键点的Z坐标
- `σ(Z_j)` 是Z坐标的标准差
- `M` 是处理的帧数

#### 3. 音视频同步性计算

**公式：**
```
Synchrony = 1 - |E_audio - E_video|
```

**数学公式：**
```
S = 1 - |E_a - E_v|
```

其中：
- `E_a` 是音频Energy
- `E_v` 是视频Energy
- `S` 是同步性分数

## 🎯 C - Cohesion（上下文一致性）

### 定义
Cohesion指标衡量参与者在会议中的状态稳定性和团队融入度，包含两个维度：
1. **个人一致性（Personal Cohesion）** - 反映状态稳定性
2. **环境一致性（Environmental Cohesion）** - 反映团队融入度

### 计算方法

#### 1. 文本一致性计算

**高级方法（使用文本嵌入）：**

1. **文本嵌入提取**
   ```python
   # 使用sentence-transformers提取文本嵌入
   embedding = text_embedding_pipeline(text)
   embeddings = np.array([embedding[0][0] for text in texts])
   ```

2. **余弦相似度计算**
   ```python
   # 计算相似度矩阵
   similarity_matrix = cosine_similarity(embeddings)
   
   # 计算平均一致性
   cohesion = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
   ```

**数学公式：**
```
C_text = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \cos(\vec{e_i}, \vec{e_j})
```

其中：
- `\vec{e_i}` 是第i个文本的嵌入向量
- `N` 是文本数量
- `\cos(\vec{e_i}, \vec{e_j})` 是余弦相似度

**简化方法（词汇重叠）：**

1. **词汇重叠计算**
   ```python
   # 计算Jaccard相似度
   words1 = set(text1.lower().split())
   words2 = set(text2.lower().split())
   
   intersection = len(words1.intersection(words2))
   union = len(words1.union(words2))
   similarity = intersection / union if union > 0 else 0.0
   ```

**数学公式：**
```
C_text = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \frac{|W_i \cap W_j|}{|W_i \cup W_j|}
```

其中：
- `W_i` 是第i个文本的词汇集合
- `|W_i \cap W_j|` 是交集大小
- `|W_i \cup W_j|` 是并集大小

#### 2. 个人一致性计算

**基于情感分析的方法：**

1. **情感分数计算**
   ```python
   # 对每个说话者的文本计算情感分数
   sentiments = []
   for text in texts:
       scores = sentiment_analyzer.polarity_scores(text)
       sentiments.append(scores['compound'])
   ```

2. **稳定性计算**
   ```python
   # 计算情感变化的标准差，标准差越小越稳定
   sentiment_std = np.std(sentiments)
   stability = 1.0 / (1.0 + sentiment_std)
   ```

**数学公式：**
```
C_personal = \frac{1}{S} \sum_{s=1}^{S} \frac{1}{1 + \sigma_s}
```

其中：
- `S` 是说话者数量
- `\sigma_s` 是第s个说话者情感分数的标准差

#### 3. 环境一致性计算

**基于情感分析的方法：**

1. **环境情感计算**
   ```python
   # 计算所有文本的平均情感
   all_sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] 
                     for text in all_texts]
   env_sentiment = np.mean(all_sentiments)
   ```

2. **说话者偏差计算**
   ```python
   # 计算每个说话者与环境的偏差
   for speaker, texts in speaker_texts.items():
       speaker_sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] 
                            for text in texts]
       speaker_avg = np.mean(speaker_sentiments)
       deviation = 1.0 - abs(speaker_avg - env_sentiment)
       speaker_deviations.append(deviation)
   ```

**数学公式：**
```
C_environmental = \frac{1}{S} \sum_{s=1}^{S} \left(1 - |\bar{s_s} - \bar{s_{env}}|\right)
```

其中：
- `\bar{s_s}` 是第s个说话者的平均情感分数
- `\bar{s_{env}}` 是环境平均情感分数

#### 4. 综合一致性计算

**公式：**
```
C_overall = 0.4 × C_personal + 0.6 × C_environmental
```

**数学公式：**
```
C = 0.4 \cdot C_p + 0.6 \cdot C_e
```

其中：
- `C_p` 是个人一致性
- `C_e` 是环境一致性
- 权重可以根据应用场景调整

## 📊 计算流程

### 完整VAEC计算流程

```python
def calculate_vaec_scores(data_path: str) -> Dict[str, float]:
    results = {
        'valence': 0.0,
        'arousal': 0.0,
        'energy': 0.0,
        'cohesion': 0.0
    }
    
    if data_path.endswith(('.wav', '.mp3', '.flac')):
        # 音频文件
        va, aa = va_calculator.calculate_audio_va(data_path)
        ea = energy_calculator.calculate_audio_energy(data_path)
        
        results['valence'] = va
        results['arousal'] = aa
        results['energy'] = ea
        
    elif data_path.endswith(('.mp4', '.avi', '.mov')):
        # 视频文件
        va, aa = va_calculator.calculate_video_va(data_path)
        ev = energy_calculator.calculate_video_energy(data_path)
        
        results['valence'] = va
        results['arousal'] = aa
        results['energy'] = ev
        
    elif data_path.endswith(('.txt', '.json')):
        # 文本文件
        vt, at = va_calculator.calculate_text_va(text)
        results['valence'] = vt
        results['arousal'] = at
    
    return results
```

## 🎯 指标解释

### Energy指标解释

| Energy分数 | 解释 | 含义 |
|------------|------|------|
| 0.0 - 0.3 | 低活跃度 | 参与者参与度较低，可能缺乏兴趣或注意力 |
| 0.3 - 0.7 | 中等活跃度 | 参与者正常参与，与会议节奏协调 |
| 0.7 - 1.0 | 高活跃度 | 参与者高度活跃，积极参与讨论 |

### Cohesion指标解释

| Cohesion分数 | 解释 | 含义 |
|--------------|------|------|
| 0.0 - 0.3 | 低一致性 | 参与者状态不稳定或与团队融入度低 |
| 0.3 - 0.7 | 中等一致性 | 参与者状态相对稳定，与团队有一定融入度 |
| 0.7 - 1.0 | 高一致性 | 参与者状态稳定，与团队高度融入 |

## 🔧 技术实现细节

### 依赖库
- **音频处理**: librosa
- **视频处理**: OpenCV, MediaPipe
- **文本处理**: sentence-transformers, NLTK
- **数学计算**: numpy, scipy

### 性能优化
1. **批处理**: 对大量数据进行批处理
2. **缓存机制**: 缓存中间计算结果
3. **并行处理**: 对独立任务进行并行计算
4. **简化回退**: 当高级方法不可用时使用简化方法

### 错误处理
1. **异常捕获**: 每个计算步骤都有异常处理
2. **默认值**: 计算失败时返回合理的默认值
3. **日志记录**: 详细记录计算过程和错误信息
4. **回退机制**: 高级方法失败时自动回退到简化方法

## 📈 应用场景

### Energy指标应用
- **会议参与度评估**: 评估参与者的活跃程度
- **演讲效果分析**: 分析演讲者的表现力
- **团队协作评估**: 评估团队成员的参与度

### Cohesion指标应用
- **团队融入度分析**: 评估新成员的融入情况
- **会议效果评估**: 评估会议的整体效果
- **个人状态监控**: 监控参与者的状态稳定性

## 🎉 总结

E、C指标为ContextVibe项目提供了重要的情感分析维度：

1. **E（Energy）** - 通过音频和视频分析评估交互同步性
2. **C（Cohesion）** - 通过文本分析评估上下文一致性

这两个指标与V（Valence）和A（Arousal）一起，构成了完整的VAEC情感分析框架，为会议情感分析提供了全面的评估工具。

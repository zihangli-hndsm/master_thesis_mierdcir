# CIR Rewrite Annotation Guideline / CIR 重写文本人工标注指南

## English Instructions

### Goal
You will evaluate whether the rewritten MiERDCIR text is a good replacement for the original MTCIR text in a Composed Image Retrieval task.

Each example shows:
- Reference Image: the starting image.
- Target Image: the desired retrieval target.
- Original Text (MTCIR): the original edit/search instruction.
- Rewritten Text (MiERDCIR): the rewritten instruction that should preserve the same intent while being useful for retrieval.

The rewritten text is shown explicitly because the main goal is to identify when the rewrite fails.

### What To Score
Use the sliders from 0 to 100. Score the Original MTCIR text and the Rewritten MiERDCIR text independently.

1. Intent Preservation
Does the text keep the core editing/search goal?
- 0: The text describes a different goal or contradicts the target.
- 50: The main goal is partly preserved but important details are missing or changed.
- 100: The core goal and important constraints are preserved.

2. Naturalness / User-likeness
Does the text sound like a natural human search or edit query?
- 0: Broken, unnatural, hard to understand.
- 50: Understandable but awkward, verbose, or machine-like.
- 100: Clear, fluent, and natural.

3. Discriminativeness for Retrieval
Is the text specific enough to help a retriever distinguish the target from similar images?
- 0: Too vague or generic to identify the target.
- 50: Some useful details, but still ambiguous.
- 100: Specific, target-relevant, and retrieval-friendly.

4. Harmful Omission or Hallucination
Did the text drop critical attributes or invent false details? For this metric, higher is worse.
- 0: No harmful omissions or hallucinations.
- 50: Some missing or questionable details.
- 100: Severe omission, contradiction, or invented detail that would mislead retrieval.

### Fail Case Checkbox
Check "Mark as a typical Fail Case for MiERDCIR" when the rewritten MiERDCIR text clearly fails. Typical failures include:
- It changes the original intent.
- It removes a critical visual attribute or constraint.
- It invents details not supported by the reference/target pair.
- It becomes too vague for retrieval.
- It is much less natural or less useful than the original text.

### Notes
Use the notes box for short comments when helpful, especially for MiERDCIR failures. For example:
- "Rewrite removes the color constraint."
- "Hallucinates a person not visible in the target."
- "Too generic to retrieve the target."

### Running The App
1. Unzip the annotation package.
2. Open a terminal in the unzipped folder.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Start the app for your assigned annotator ID:

```bash
./run_annotator_1.sh
```

or

```bash
./run_annotator_2.sh
```

5. Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

6. Click "Submit Annotation" after each example. Your results are saved automatically as:

```text
evaluation_results_Annotator_1.csv
```

or

```text
evaluation_results_Annotator_2.csv
```

Send back only your completed CSV file unless instructed otherwise.

### Important Rules
- Do not edit `samples.json`.
- Do not rename the generated CSV file.
- If you stop and restart the app, it will continue from the first unfinished sample.
- If an image is missing, write a note and continue if possible.

---

## 中文说明

### 标注目标
你需要评估 MiERDCIR 重写文本是否可以作为原始 MTCIR 文本的高质量替代，用于组合图像检索任务。

每个样本包含：
- Reference Image：起始参考图像。
- Target Image：希望检索到的目标图像。
- Original Text (MTCIR)：原始编辑/检索指令。
- Rewritten Text (MiERDCIR)：重写后的指令，应当保持原始意图，并且对检索有帮助。

界面会明确显示哪一个是重写文本，因为本次标注的重点是判断重写文本什么时候失败。

### 评分指标
所有滑动条范围为 0 到 100。请分别为 Original MTCIR 文本和 Rewritten MiERDCIR 文本打分。

1. Intent Preservation（意图保持）
文本是否保留了核心编辑/检索目标？
- 0：文本描述了不同目标，或与目标图像矛盾。
- 50：主要目标部分保留，但重要细节缺失或改变。
- 100：核心目标和重要约束都被很好保留。

2. Naturalness / User-likeness（自然度 / 用户表达相似度）
文本是否像真实用户会写出的搜索或编辑请求？
- 0：表达破碎、不自然、难以理解。
- 50：可以理解，但生硬、啰嗦或很像机器生成。
- 100：清晰、流畅、自然。

3. Discriminativeness for Retrieval（检索区分度）
文本是否足够具体，能帮助检索器从相似图像中找到目标图像？
- 0：过于模糊或泛泛而谈，无法定位目标。
- 50：包含一些有用信息，但仍有明显歧义。
- 100：具体、与目标相关，并且对检索非常有帮助。

4. Harmful Omission or Hallucination（有害遗漏或幻觉）
文本是否遗漏了关键属性，或编造了不存在的细节？注意：这个指标分数越高表示越差。
- 0：没有有害遗漏或幻觉。
- 50：存在一些缺失或可疑细节。
- 100：严重遗漏、矛盾或编造信息，会明显误导检索。

### Fail Case 复选框
当 MiERDCIR 重写文本明显失败时，请勾选 "Mark as a typical Fail Case for MiERDCIR"。常见失败包括：
- 改变了原始意图。
- 删除了关键视觉属性或约束。
- 编造了参考图或目标图中没有的细节。
- 变得过于笼统，无法支持检索。
- 相比原文明显更不自然或更不适合检索。

### Notes 备注
如果有必要，请在 notes 中简短说明原因，尤其是 MiERDCIR 失败样本。例如：
- "Rewrite removes the color constraint."（重写删除了颜色约束。）
- "Hallucinates a person not visible in the target."（编造了目标图中不存在的人。）
- "Too generic to retrieve the target."（过于笼统，无法检索目标。）

### 如何运行程序
1. 解压标注压缩包。
2. 在解压后的文件夹中打开终端。
3. 安装依赖：

```bash
python -m pip install -r requirements.txt
```

4. 根据分配给你的标注员 ID 启动程序：

```bash
./run_annotator_1.sh
```

或

```bash
./run_annotator_2.sh
```

5. 打开终端中显示的本地网址，通常是：

```text
http://127.0.0.1:7860
```

6. 每个样本完成后点击 "Submit Annotation"。结果会自动保存为：

```text
evaluation_results_Annotator_1.csv
```

或

```text
evaluation_results_Annotator_2.csv
```

除非另有说明，完成后只需要发回生成的 CSV 文件。

### 重要规则
- 不要修改 `samples.json`。
- 不要重命名生成的 CSV 文件。
- 如果中途关闭程序，再次启动后会从第一个未完成样本继续。
- 如果图片缺失，请在 notes 中说明，并在可行情况下继续标注。

# 2.3 标注练习（X-AnyLabeling）

对应《视觉教材 YOLO 部分》2.3.2 标注工具、2.3.4 标签文件格式。

## 目录结构

    label_practice/
    ├── images/
    │   ├── 2.3_label_practice_01.jpg
    │   └── 2.3_label_practice_01.json   # 工具原生标注，必须与图同名同目录
    ├── labels/
    │   └── 2.3_label_practice_01.txt    # 导出的 YOLO 标签
    ├── classes.txt                      # 类别表，顺序 = 类别下标
    └── README.md

## 类别表（顺序不可改）

| 下标 | 名称 |
| --- | --- |
| 0 | baizi |
| 1 | cheli |
| 2 | shuiguo |

改顺序会让所有标签错位，`cheli` 就会变成别的类。

## 重新导出

1. 在 X-AnyLabeling 里打开 `images/` 目录，标注完成后保存；
2. 导出选 YOLO(HBB) 格式，弹出 **Select a specific classes file** 时，
   **必须选本目录的 `classes.txt`**——选空文件会导致导出的 txt 为空；
3. 导出路径保持默认 `../labels`，正好落到 `label_practice/labels/`；
4. 导出后确认 txt 里有一行 `1 0.427920 0.546029 0.300409 0.227355`。

没有 `.json` 就导不出东西：工具靠"图名.json"去找标注。

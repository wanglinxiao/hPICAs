## Description
我们提供了文章中部分方法的示例代码，包括构建跨物种预测模型、CNN的可解释性分析、鉴定hPICA区域、以及基于CNN来预测变异的遗传效应。我们也提供了必要的示例数据，以便其他用户可以实现我们的代码。同时，我们也将训练好的模型存储在google drive。
#### 评测CNN的跨物种预测能力
`https://drive.google.com/file/d/1-xiKOT68UPYMcJElYzByrrKfnUr4zl7k/view?usp=drive_link`
#### 成年人111个细胞类型的预测模型（single-cell ATAC-seq）
`https://drive.google.com/file/d/1A4CNxeXiSGZYTylL-ORweKgMBMvpdHym/view?usp=drive_link`

## Tutorials
### CNN的跨物种预测评估
为了评估CNN在灵长类动物的跨物种预测染色质开放性的表现，我们收集了淋巴细胞系、脑额前叶以及大脑brodmann area的数据集，数据集包含人、黑猩猩以及猕猴。通过比较跨物种预测的AUROC以及输入序列motif类型的相似性，来说明CNN具有robust跨物种预测表现。代码详情见folder CNN

### 鉴定hPICA
<img width="536" alt="屏幕截图 2025-05-23 153411" src="https://github.com/user-attachments/assets/17573456-c1fa-45e6-a699-b31e3fa52433" />

# 1. Metric Measurements
The following is a table of the metric measurements of Llama 2, Phi-2, and Mistral 7B fine-tuned on the [flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) dataset:

| Model       | BLEU  | Rouge-L | BERTScore | CodeBLEU | Human Evaluation |
|-------------|-------|---------|-----------|----------|------------------|
| Llama 2     | 0.85  | 0.83    | 0.86      | 0.93     | 1.00             |
| Phi-2       | 0.74  | 0.70    | 0.72      | 0.81     | 0.80             |
| Mistral 7B  | 0.86  | 0.82    | 0.84      | 0.90     | 1.00             |

In terms of overall evaluation, Llama 2 performed the best, Mistral 7B also performs very well, whereas Phi-2 is the least successful. The CodeBLEU metric appears to be the most appropriate metric for this task, as it takes into consideration of identifiers and other programming language concepts. BLEU is related to CodeBLEU and its metrics are not too far behind CodeBLEU. For human evaluation, Llama 2 and Mistral 7B produced completely accurate code, whereas Phi-2 produced 16/20 correct test samples.

# 2. Hyperparameter Tuning
The following is a table show the impact of various hyperparameter values of Llama 2, Phi-2, and Mistral 7B fine-tuned on the [flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) dataset:

| Model       | Top K | Beam Size | Temperature | BLEU  | Rouge-L | BERTScore | CodeBLEU | Human Evaluation |
|-------------|-------|-----------|-------------|-------|---------|-----------|----------|------------------|
| Llama 2     | 30    | 3         | 0.70        | 0.85  | 0.83    | 0.86      | 0.93     | 1.00             |
| Phi-2       | 10    | 3         | 0.50        | 0.77  | 0.72    | 0.75      | 0.78     | 0.85             |
| Mistral 7B  | 15    | 4         | 0.50        | 0.85  | 0.83    | 0.83      | 0.86     | 0.95             |

When tuning the hyperparameters of the models, the top k and temperature hyperparameters had a large effect on the generated output. To help Phi-2 have more optimal results, the top k was lowered to 10 and temperature was lowered to 0.5 to produce more conservative results, allowing it to introduce less irrelevant code. A similar tuning was done for Mistral 7B, however, this had the opposite effect where the model performed worse. To help explore more potential code sequences, the beam size was raised from 3 to 4 which slightly helped improve results. Llama 2 performed well at the default hyperparameter values.
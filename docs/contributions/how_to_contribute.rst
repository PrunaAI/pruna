Since Pruna offers a broad range of compression algorithms, the following table provides a high-level overview of all methods available in Pruna. For a detailed description of each algorithm, have a look at our [documentation](https://docs.pruna.ai/en/stable/).


| Technique | Description | Speed | Memory | Accuracy |
| --- | --- | --- | --- | --- |
| Batching | Groups multiple inputs together to be processed simultaneously, improving computational efficiency and reducing overall processing time. | ✅ | ❌ | 〰️ |
| Caching | Stores intermediate results of computations to speed up subsequent operations, reducing inference time by reusing previously computed results. | ✅ | 〰️ | 〰️ |
| Compilation | Compilation optimises the model with instructions for specific hardware. | ✅ | ➖ | 〰️ |
| Distillation | Trains a smaller, simpler model to mimic a larger, more complex model. | ✅ | ✅ | ❌ |
| Quantization | Reduces the precision of weights and activations, lowering memory requirements. | ✅ | ✅ | ❌ |
| Pruning | Removes less important or redundant connections and neurons, resulting in a sparser, more efficient network. | ✅ | ✅ | ❌ |
| Recovering | Restores the performance of a model after compression. | 〰️ | 〰️ | ✅ |

✅(improves), ➖(stays the same), 〰️(could worsen), ❌(worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## <img src="./docs/assets/images/pruna_sad.png" alt="Pruna Sad" width=20></img> FAQ and Troubleshooting

If you can not find an answer to your question or problem in our [documentation][documentation], in our [FAQs][docs-faq] or in an existing issue, we are happy to help you! You can either get help from the Pruna community on [Discord][discord], join our [Office Hours][docs-office-hours] or open an issue on GitHub.
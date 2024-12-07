## Investigating Attention Mechanisms in LLMs for Downstream Applications

### Model Fine-tuning

The scripts to fine-tune models to the sentiment classification task are ``finetune/xxx_classifier_imdb_finetune.py``.

The scripts to fine-tune models to the text summarization task are ``finetune/xxx_summarization_cnn_finetune.py``.

### Comparisons of Attention Patterns

The scripts to compare attention scores before and after fine-tuning are ``comparison_attention/compare_vis_imdb.ipynb`` and 
``comparison_attention/compare_vis.ipynb``.

### Analysis of [CLS] Token Attention in Sentiment Classification

The scripts are available at the ``imbd_attention_analysis/scripts`` folder. 

We collected the attention scores, sentiment scores and token labels of the content tokens with the maximum attention scores at each layer.
The scripts we used are ``collect_attention_bert.py``, ``collect_attention_bigbird.py``, and ``collect_attention_longformer.py``.

We also collected the attention distribution of those max tokens over layers, using scripts
``collect_cls_attention_xxx.py`` (for BERT, BigBird, and LongFormer).

The analysis jupyter notebooks are at ``imbd_attention_analysis/analysis`` folder. We explored the BERT model and explored the attentions mechanism
using ``bert_example.ipynb`` and  ``explore_attention_over_layer.ipynb``. We analyzed the results using ``imbd_analysis.ipynb``.

Finally, the figures are stored at ``imbd_attention_analysis/figures``.

### Experimental Results on CNN/DailyMail Fine-tuned LLMs

The scripts are available at the ``cnn_attention_analysis/scripts`` folder. 

In particular, we analyzed the results using ``xxx_summarization_cnn_finetune_analysis.py`` for the three models. We looked at the ground truth highlights and predicted highlights of the articles. The correlation between attention scores and output imprtance scores are saved the ``./ckpts/`` folder.

### Exploration

We saved the exploratory analyses at the ``exploration`` folder.
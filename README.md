## Aspect Level Sentiment Classification with Deep Memory Network

[TensorFlow](https://www.tensorflow.org/) implementation of [Tang et al.'s EMNLP 2016](https://arxiv.org/abs/1605.08900) work.

### Problem Statement
Given a sentence and an aspect occurring in the sentence, this task aims at inferring the sentiment polarity (e.g. positive, negative, neutral) of the aspect.

### Example
For example, in sentence ''great food but the service was dreadful!'', the sentiment polarity of aspect ''food'' is positive while the polarity of aspect ''service'' is negative.

### Quick Start
Download the 300-dimensional pre-trained word vectors from [Glove](http://nlp.stanford.edu/projects/glove/) and save it in the 'data' folder as 'data/glove.6B.300d.txt'. 

Train a model with 7 hops on the [Laptop](http://alt.qcri.org/semeval2016/task5/) dataset.
```
python main.py --show True
```

Note this code requires TensorFlow, Future and Progress packages to be installed.

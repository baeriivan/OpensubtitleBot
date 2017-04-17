## Retrieval-Based Conversational Model in Tensorflow (Ubuntu Dialog Corpus)

#### Overview

This is an implementation of the Dual LSTM Encoder model for the [OpenSubtitle dataset](http://opus.lingfil.uu.se/OpenSubtitles2013.php).

It is an adaptation from this [chatbot project](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/), where some updates have been made to be compatible with TF 1.0, and with additionnal preprocessing on the data to be able to handle the collection of xmls given by OpenSubtitle.

Assumption with the data: we consider each line as being consecutively a question and an answer.

This is only a first draft on the project. Much more pre-processing of the data can be done, and parameter better tuned.

#### Get the Data

Download the [OpenSubtitle dataset](http://opus.lingfil.uu.se/OpenSubtitles2013.php) and extract the archive into `./data`.


### Prepare the Data

After having changed some paths and parameters in the script, lauch:

```
python3 xml2csv.py
python3 prepare_data.py
```

#### Training

```
python3 rb_train.py
```


#### Evaluation

```
python3 rb_test.py --model_dir=...
```


#### Evaluation

```
python3 rb_predict.py --model_dir=...
```

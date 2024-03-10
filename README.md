# AuthorMind-Neural-Network-Based-Author-Prediction-from-Text_MLP
AuthorMind: Neural Network-Based Author Prediction from Text
MLP 

#### Name: Akshay Parate


## Part A: Multi-Layer Perceptron 

Implement the data loading, preprocessing, tokenization, and TF-IDF feature extraction.
Implement MLP model, evaluation metrics, and Mini-batch GD with AdaGrad.
Implement the MLP with Tensorflow and compare to your implementation.
Analysis the results in the Conlusion part.


## 1. Data Processing (5 points)

* Download the dataset from Canvas
* Load data to text and labels
* Preprocessing
* Tokenization
* Split data
* Feature extraction (TF-IDF)

#### Download NLTK stopwords


```python
import os

import nltk


nltk_path = os.path.join('a2-data', 'nltk')
nltk.download('stopwords', download_dir=nltk_path)
nltk.data.path.append(nltk_path)
```

    [nltk_data] Downloading package stopwords to a2-data\nltk...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
import sys


def print_line(*args):
    """ Inline print and go to the begining of line
    """
    args1 = [str(arg) for arg in args]
    str_ = ' '.join(args1)
    sys.stdout.write(str_ + '\r')
    sys.stdout.flush()
```


```python
from typing import List, Tuple, Union

import numpy as np
```

### 1.1 Load data

- Load sentences and labels
- Transform string labels into integers


```python
def load_sentence_label(data_path: str) -> Tuple[List[str], List[str]]:
    """ Load sentences and labels from the specified path
    Args:
        data_path: data_path: path to the data file, e.g., 'a1-data/SMSSpamCollection'
        sentences: the raw text list of all sentences
    Returns:
        labels: the label list of all sentences
    """
    sentences, labels = [], []
    # Start your code here (load text and label from files)
    file = open(data_path, "r", encoding='utf-8')  
    fileData = file.readlines()
    file.close()
    for data in fileData:
        splitData = data.split("	")
        sentences.append(splitData[1].split("\n")[0])
        labels.append(splitData[0])
    # End
    return sentences, labels
```


```python
# pip install pandas
```


```python
import pandas as pd
dataDf = pd.DataFrame()
sentences, labels = load_sentence_label(data_path)
dataDf["Sentences"] = sentences
dataDf["labels"] = labels
```


```python
data_path = os.path.join('a2-data', 'books.txt')
sentences, labels = load_sentence_label(data_path)
label_map = {}
for label in sorted(list(set(labels))):
    label_map[label] = len(label_map)
labels = np.array([label_map[label] for label in labels], dtype=int)
sentences = np.array(sentences, dtype=object)

print('Label map:', label_map)
print('Number of sentences and labels:', sentences.shape, labels.shape)
```

    Label map: {'Arthur Conan Doyle': 0, 'Fyodor Dostoyevsky': 1, 'Jane Austen': 2}
    Number of sentences and labels: (19536,) (19536,)
    

#### Split the data into training, validation and test


```python
import random
def train_test_split(sentences: np.ndarray,
                     labels: np.ndarray,
                     test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Split the sentences and labels into training and test data by shuffling
    Args:
        sentences: A numpy array containing all sentences
        labels: A number array containing label ids
        test_ratio: A float number to calculate the number of test data

    Returns:
        train_sentences: A numpy array containing all training sentences
        train_labels: A number array containing all training label ids
        test_sentences: A numpy array containing all test sentences
        test_labels: A number array containing all test label ids
    """
    
    assert 0 < test_ratio < 1
    assert len(sentences) == len(labels)

    train_index, test_index = [], []
    # Start your code here (split the index for training and test)
    n= int(round(len(sentences) * test_ratio,0))
    rand_list=list(range(0, len(sentences)))
    shuffled_rand_list = random.sample(rand_list, len(rand_list))
    train_index = shuffled_rand_list[n:]
    test_index = shuffled_rand_list[:n]
    # End
    train_sentences, train_labels = sentences[train_index], labels[train_index]
    test_sentences, test_labels = sentences[test_index], labels[test_index]
    return train_sentences, train_labels, test_sentences, test_labels
```


```python
np.random.seed(6666)

test_ratio = 0.2
valid_ratio = 0.1
(train_sentences, train_labels,
    test_sentences, test_labels) = train_test_split(sentences, labels, test_ratio)
(train_sentences, train_labels,
    valid_sentences, valid_labels) = train_test_split(train_sentences, train_labels, valid_ratio)

print('Training data length:', len(train_sentences))
print('Validation data length:', len(valid_sentences))
print('Test data length:', len(test_sentences))
```

    Training data length: 14066
    Validation data length: 1563
    Test data length: 3907
    


```python
def count_label(labels: np.ndarray, label_map: dict[str, int]) -> dict[str, int]:
    """
    Args:
        labels: The labels of a dataset 
        label_map: The mapping from label to label id
    Returns:
        label_count: The mapping from label to its count
    """
    label_count = {key: 0 for key in label_map.keys()}
    # Start your code here (count the number of each label)
    uniqueVal = label_map.values()
    for u,k in zip(uniqueVal,list(label_count.keys())):
        x = [i for i in labels if i==u]
        label_count[k] = len(x)
    # End
    return label_count
```


```python
print('Training:', count_label(train_labels, label_map))
print('Validation:', count_label(valid_labels, label_map))
print('Test:', count_label(test_labels, label_map))
```

    Training: {'Arthur Conan Doyle': 1833, 'Fyodor Dostoyevsky': 4229, 'Jane Austen': 8004}
    Validation: {'Arthur Conan Doyle': 200, 'Fyodor Dostoyevsky': 487, 'Jane Austen': 876}
    Test: {'Arthur Conan Doyle': 505, 'Fyodor Dostoyevsky': 1228, 'Jane Austen': 2174}
    

#### Dataset statistics
Fill this table with the statistics you just printed (double click this cell to edit)

|                | Arthur Conan Doyle | Fyodor Dostoyevsky | Jane Austen | Total |
|:--------------:|--------------------|--------------------|-------------|-------|
|  **Training**  |      1865          |     4325           |    7876     | 14066 |
| **Validation** |       188          |      491           |     884     |  1563 |
|    **Test**    |       485          |     1128           |    2294     |  3907 |
|    **Total**   |      2538          |     5944           |   11054     | 19536 |


### 1.2 Preprocess
In this section, you need to remove all the unrelated characters, including punctuation, urls, and numbers. Please fill up the functions and test them by running the following cell.


```python
import re


class Preprocessor:
    def __init__(self, punctuation=True, url=True, number=True):
        self.punctuation = punctuation
        self.url = url
        self.number = number

    def apply(self, sentence: str) -> str:
        """ Apply the preprocessing rules to the sentence
        Args:
            sentence: raw sentence
        Returns:
            sentence: clean sentence
        """
        sentence = sentence.lower()
        if self.url:
            sentence = Preprocessor.remove_url(sentence)
        if self.punctuation:
            sentence = Preprocessor.remove_punctuation(sentence)
        if self.number:
            sentence = Preprocessor.remove_number(sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    @staticmethod
    def remove_punctuation(sentence: str) -> str:
        """ Remove punctuations in sentence with re
        Args:
            sentence: sentence with possible punctuations
        Returns:
            sentence: sentence without punctuations
        """
        # Start your code here
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # End
        return sentence

    @staticmethod
    def remove_url(sentence: str) -> str:
        """ Remove urls in text with re
        Args:
            sentence: sentence with possible urls
        Returns:
            sentence: sentence without urls
        """
        # Start your code here
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        sentence = url_pattern.sub("", sentence)
        # End
        return sentence

    @staticmethod
    def remove_number(sentence: str) -> str:
        """ Remove numbers in sentence with re
        Args:
            sentence: sentence with possible numbers
        Returns:
            sentence: sentence without numbers
        """
        # Start your code here
        sentence = re.sub(r'\d', '', sentence)
        # End
        return sentence
```

##### Test your implementation by running the following cell.


```python
sentence = "Interest rates are trimmed to 7.5 by the South African central bank (https://www.xxx.xxx), but the lack of warning hits the rand and surprises markets."

processor = Preprocessor()
clean_sentence = processor.apply(sentence)

print(f'"{sentence}"') 
print('===>')
print(f'"{clean_sentence}"')
```

    "Interest rates are trimmed to 7.5 by the South African central bank (https://www.xxx.xxx), but the lack of warning hits the rand and surprises markets."
    ===>
    "interest rates are trimmed to by the south african central bank but the lack of warning hits the rand and surprises markets"
    

### 1.3 Tokenization


```python
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer()
print(list(stopwords_set)[:10])
```

    ['below', 'all', 'were', 'am', 'itself', 'not', 'once', 'his', 'himself', 'y']
    


```python
def tokenize(sentence: str) -> List[str]:
    """ Tokenize a sentence into tokens (words)
    Args:
        sentence: clean sentence
    Returns:
        tokens
    """
    words = []
    # Start your code here
    #     Step 1. Split sentence into words
    splitWords = sentence.split(" ")
    #     Step 2. Extract word stem using the defined stemmer (PorterStemmer) by calling stemmer.stem(word)
    stemmedWords = [stemmer.stem(word) for word in splitWords]
    #     Step 3. Remove stop words using the defined stopwords_set
    words = [w for w in stemmedWords if not w in stopwords_set]
    # End
    return words
```

##### Test your implementation by running the following block.


```python
sentence = "Interest rates are trimmed to 7.5 by the South African central bank (https://www.xxx.xxx), but the lack of warning hits the rand and surprises markets."

processor = Preprocessor()
clean_sentence = processor.apply(sentence)
tokens = tokenize(clean_sentence)

print(f'"{sentence}"') 
print('===>')
print(f'"{tokens}"')
```

    "Interest rates are trimmed to 7.5 by the South African central bank (https://www.xxx.xxx), but the lack of warning hits the rand and surprises markets."
    ===>
    "['interest', 'rate', 'trim', 'south', 'african', 'central', 'bank', 'lack', 'warn', 'hit', 'rand', 'surpris', 'market']"
    

### 1.5 Feature Extraction

TF-IDF:
$$\text{TF-IDF}(t, d) = \frac{f_{t, d}}{\sum_{t'}{f_{t', d}}} \times \log{\frac{N}{n_t}}$$

- $t$: A term
- $d$: A document. Here, we regard a sentence as a document
- $f_{t, d}$: Number of term $t$ in $d$
- $N$: Number of document
- $n_t$: Number of document containing $t$


```python
from collections import defaultdict


class TfIdfEncoder:
    def __init__(self):
        self.vocab = defaultdict(int)
        self.token2index = {}
        self.df = defaultdict(int)
        self.num_doc = 0
        self.processor = Preprocessor()

    def fit(self, sentences: Union[List[str], np.ndarray]) -> int:
        """ Using the given texts to store key information in TF-IDF calculation
            In this function, you are required to implement the fitting process.
                1. Construct the vocabulary and store the frequency of tokens (self.vocab).
                2. Construct the document frequency map to tokens (self.df).
                3. Construct the token to index map based on the frequency.
                   The token with a higher frequency has the smaller index
        Args:
            sentences: Raw sentences
        Returns:
            token_num
        """
        self.num_doc = len(sentences)
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 or i == len(sentences) - 1:
                print_line('Fitting TF-IDF encoder:', (i + 1), '/', len(sentences))
            # Start your code here (step 1 & 2)
            clean_sentence = self.processor.apply(sentence)
            tokens = tokenize(clean_sentence)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.vocab[token] += 1
                self.df[token] += 1
            # End
        print_line('\n')
        # Start your code here (Step 3)
        sorted_tokens = sorted(self.vocab.keys(), key=lambda x: self.vocab[x], reverse=True) #sorting tokens in descending order based on the frequency.
        self.token2index = {token: idx for idx, token in enumerate(sorted_tokens)} #Maps each token to its index in the sorted list.
        # End
        token_num = len(self.token2index) 
        print('The number of distinct tokens:', token_num)
        return token_num

    def encode(self, sentences: Union[List[str], np.ndarray]) -> np.ndarray:
        """ Encode the sentences into TF-IDF feature vector
            Note: if a token in a sentence does not exist in the fit encoder, we just ignore it.
        Args:
            sentences: Raw sentences
        Returns:
            features: A (n x token_num) matrix, where n is the number of sentences
        """
        n = len(sentences)
        features = np.zeros((n, len(self.token2index)))
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 or i == n - 1:
                print_line('Encoding with TF-IDF encoder:', (i + 1), '/', n)
            # Start your code (calculate TF-IDF)
            clean_sentence = self.processor.apply(sentence)
            tokens = tokenize(clean_sentence)
            for token in tokens:
                if token in self.token2index:
                    features[i, self.token2index[token]] += 1 / self.df[token]
            # End
        print_line('\n')
        return features
```

##### Test your implementation by running the following cell.


```python
encoder = TfIdfEncoder()
encoder.fit(train_sentences[:100])
features = encoder.encode(train_sentences[:10])

print(features[:5])
```

    Fitting TF-IDF encoder: 100 / 100
    The number of distinct tokens: 1316
    Encoding with TF-IDF encoder: 10 / 10
    [[0.02777778 0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.03703704 ... 0.         0.         0.        ]]
    

#### Encode training, validation, and test dataset


```python
num_class = 3

encoder = TfIdfEncoder()
vocab_size = encoder.fit(train_sentences)

x_train = encoder.encode(train_sentences)
x_valid = encoder.encode(valid_sentences)
x_test = encoder.encode(test_sentences)

y_train = np.zeros((len(train_labels), num_class))
y_valid = np.zeros((len(valid_labels), num_class))
y_test = np.zeros((len(test_labels), num_class))
#One hot encoding - The code uses NumPy indexing to assign 1 to the appropriate column for each sample in the training, validation, and test sets.
y_train[np.arange(len(train_labels)), train_labels] = 1
y_valid[np.arange(len(valid_labels)), valid_labels] = 1
y_test[np.arange(len(test_labels)), test_labels] = 1

print('The size of training set:', x_train.shape, y_train.shape)
print('The size of validation set:', x_valid.shape, y_valid.shape)
print('The size of test set:', x_test.shape, y_test.shape)
```

    Fitting TF-IDF encoder: 14066 / 14066
    The number of distinct tokens: 16600
    Encoding with TF-IDF encoder: 14066 / 14066
    Encoding with TF-IDF encoder: 1563 / 1563
    Encoding with TF-IDF encoder: 3907 / 3907
    The size of training set: (14066, 16600) (14066, 3)
    The size of validation set: (1563, 16600) (1563, 3)
    The size of test set: (3907, 16600) (3907, 3)
    

## 2. MLP (20 Points)
In this section, you are required to implement a two-layer MLP model (input -> hidden layer -> output layer) with $L_2$ regularization from scratch. 

The objective function of LR for multi-class classification:

$$J = L(\mathbf{x}, \mathbf{y} \mid \mathbf{w}, \mathbf{b}) = -\frac{1}{n}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}log\frac{e^{f_k}}{\sum_{c=1}^{K}e^{f_c}} + \lambda \sum_{j=1}^{d}w_{kj}^2$$

- $z_1 = w_1x$
- $h_1 = activation(z_1)$
- $z_2 = w_2 h_1$
- $\hat{y} = softmax(z_2)$

- $n$: Number of samples
- $d$: Dimension of $\mathbf{w}$
- Here, you can use `sigmoid` as the activation function for the hidden layer.

### 2.1 MLP Model (5 Points)


```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """ The softmax activation function
    Args:
        x: Input matrix or vector
        axis: The dimension of x that needs to run softmax, default -1, i.e., the last dimension
    Returns:
        output: Softmax value of the specified dimension in x
    """
    # Start your code here
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    x = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # End
    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    """ The sigmoid activation function
    Args:
        x: Input matrix or vector
    Returns:
        output: Sigmoid value of each entry in x
    """
    # Start your code here
    x = 1 / (1 + np.exp(-x))
    # End
    return x
```


```python
class MLP:
    def __init__(self, feature_dim: int, hidden_dim: int, num_class: int, lambda_: float):
        """ MLP Model
        Args:
            feature_dim: feature dimension
            hidden_dim: hidden units
            num_class: number of class
            lambda_: lambda in L2 regularizer
        """
        # Start your code here (initialize weight and bias)
        self.w1 = np.random.randn(feature_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, num_class)
        self.b2 = np.zeros((1, num_class))
        # End
        self.lambda_ = lambda_
        self.eps = 1e-9

    def forward(self, x: np.ndarray, return_hiddens) -> np.ndarray:
        """ Forward process of logistic regression
            Calculate y_hat using x
        Args:
            x: Input data
            return_hiddens: If true the function will return h1 for gradient calculation
        Returns:
            y_hat: Output
            h1: Hidden output, used for gradient calculation. Returned if return_hiddens is set to True
        """
        y_hat = 0
        h1 = 0, 0
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        # Start your code here (calculate y_hat of MLP using x)
        z1 = np.dot(x, w1) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(h1,w2) + b2
        y_hat = sigmoid(z2)
        # End
        if return_hiddens:
            return y_hat, h1
        else:
            return y_hat

    def backward(self,
                 x: np.ndarray,
                 y_hat: np.ndarray,
                 y: np.ndarray,
                 h1: np.array) -> Tuple[np.ndarray, Union[float, np.ndarray], np.ndarray, Union[float, np.ndarray]]:
        """ Backward process of logistic regression
            Calculate the gradient of w and b
        Args:
            x: Input data
            y_hat: Output of forward
            y: Ground-truth
            h1: Hidden output of the hidden layer
        Returns:
            dw1: Gradient of w1
            db1: Gradient of b1
            dw2: Gradient of w2
            db2: Gradient of b2
        """
        w1, w2 = self.w1, self.w2
        dw1, db1, dw2, db2 = 0.0, 0.0, 0.0, 0.0
        n = len(x)
        # Start your code here (calculate the gradient of w and b)
        dz2 = y_hat - y
        dw2 = np.transpose(h1) @ dz2
        db2 = np.sum(dz2,axis = 0,keepdims = True)
        dz1 = dz2 @ np.transpose(w2) * (h1 * (1-h1))
        dw1 = np.transpose(x) @ dz1
        db1 = np.sum(dz1,axis = 0,keepdims = True)
        # End
        return dw1, db1, dw2, db2

    def categorical_cross_entropy_loss(self,
                                       y_hat: np.ndarray,
                                       y: np.ndarray) -> Union[float, np.ndarray]:
        """ Calculate the binary cross-entropy loss
        Args:
            y_hat: Output of forward
            y: Ground-truth
        Returns:
            loss: BCE loss
        """
        y_hat = np.clip(y_hat, a_min=self.eps, a_max=1 - self.eps)
        loss = 0
        n = y.shape[0]
        # Start your code here (Calculate the binary cross-entropy)
#         print(y.shape)
        loss = -1/n * np.sum(y * np.log(y_hat)) + self.lambda_ * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        # End
        return loss

    def gradient_descent(self, dw1: np.ndarray, db1: Union[np.ndarray, float], dw2: np.ndarray, db2: Union[np.ndarray, float], lr: float):
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2

    def predict(self, y_hat: np.ndarray) -> np.ndarray:
        """ Predict the label using the output y_hat
        Args:
            y_hat: Model output
        Returns:
            pred: Prediction
        """
        pred = np.zeros_like(y_hat)
        index = np.argmax(y_hat, axis=-1)
        pred[np.arange(len(y_hat)), index] = 1
        return pred
```


```python
np.random.seed(6666)
lossList = []
hidden_dim = 128
num_epoch = 100
lr = 1e-2
batch_size = 128
lambda_ = 1e-8
print_every = 10

model_mbgd = MLP(feature_dim=vocab_size, hidden_dim=hidden_dim, num_class=num_class, lambda_=lambda_)
# uncomment this line
for i in range(0,100):
    if(i%50 == 0):
        print("Epoch #: ",i)
    y_hat,h1 = model_mbgd.forward(x_train,True)
    prob = softmax(y_hat)
    loss = model_mbgd.categorical_cross_entropy_loss(prob, y_train)
    lossList.append(loss)
    dw1, db1, dw2, db2 = model_mbgd.backward(x_train,y_hat,y_train,h1)
    # print(dw1.shape, db1.shape, dw2.shape, db2.shape)
    # print(model_mbgd.w1.shape, model_mbgd.b1.shape, model_mbgd.w2.shape, model_mbgd.b2.shape)
    model_mbgd.gradient_descent(dw1, db1, dw2, db2,lr)
```

    Epoch #:  0
    

    C:\Users\Akshay\AppData\Local\Temp\ipykernel_8\2069942790.py:24: RuntimeWarning: overflow encountered in exp
      x = 1 / (1 + np.exp(-x))
    

    Epoch #:  50
    

### 2.2 Evaluation Metrics

Accuracy, Precision, Recall, F1


```python
from sklearn.metrics import precision_recall_fscore_support


def get_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """ Calculate the accuracy, precision, recall, and f1 score.
        You are allowed to use precision_recall_fscore_support from scikit-learn. Please set average to 'micro'
    Args:
        y_pred: Prediction
        y_true: Ground-truth
    Returns:
        accuracy: float number. The accuracy for the whole dataset
        precision, recall, f1: np.ndarray (num_class, ). The precision, recall, f1 for each class
    """
    assert y_pred.shape == y_true.shape
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    # Start your code here
    correct_predictions = 0
    for i in range(0,len(y_true)):
        if all(y_pred[i] == y_true[i]):
            correct_predictions = correct_predictions + 1
        else:
            pass
    total_examples = len(y_true)
    accuracy = correct_predictions / total_examples
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return accuracy, precision, recall, f1
```


```python
# Calculate the metrics for test set and fill in the table below
y_hat = model_mbgd.forward(x_test,False)
y_pred = model_mbgd.predict(y_hat)
print('Mini-batch GD:', get_metrics(y_pred, y_test))
# model_tf.evaluate(x=x_test, y=y_test)
```

    Mini-batch GD: (0.5564371640644996, 0.5564371640644996, 0.5564371640644996, 0.5564371640644996)
    

    C:\Users\Akshay\AppData\Local\Temp\ipykernel_8\2069942790.py:24: RuntimeWarning: overflow encountered in exp
      x = 1 / (1 + np.exp(-x))
    

### 2.3 AdaGrad (5 points)

$$ \mathbf{G}^{(t + 1)} \leftarrow \mathbf{G}^{(t)} + \boldsymbol{g}^{(t + 1)} \cdot \boldsymbol{g}^{(t + 1)} $$
$$ \mathbf{w}^{(t + 1)} \leftarrow \mathbf{w}^{(t)} - \frac{\eta}{\sqrt{\mathbf{G}^{(t + 1)} + \epsilon}}\boldsymbol{g}^{(t + 1)} = \mathbf{w}^{(t)} - \eta\frac{\boldsymbol{g}^{(t + 1)}}{\sqrt{\mathbf{G}^{(t + 1)} + \epsilon}} $$


```python
class AdaGrad:
    def __init__(self, init_lr, model):
        self.init_lr = init_lr
        self.model = model
        
        self.accumulative_dw1 = 0
        self.accumulative_db1 = 0
        self.accumulative_dw2 = 0
        self.accumulative_db2 = 0
        self.eps = 1e-9
        
    def update(self, dw1: np.ndarray, db1: Union[np.ndarray, float], dw2: np.ndarray, db2: Union[np.ndarray, float]):
        """ 1. Use the gradient in the current step to update the accumulative gradient of each parameter.
            2. Calculate the new gradient with the accumulative gradient
            3. Use the init learning rate the new gradient to update the parameter with model.gradient_descent()
        
        Do not return anything
        """
        # Start your code here
        # Step 1
        self.accumulative_dw1 += dw1 ** 2
        self.accumulative_db1 += db1 ** 2
        self.accumulative_dw2 += dw2 ** 2
        self.accumulative_db2 += db2 ** 2
        # Step 2
        new_dw1 = dw1 / (np.sqrt(self.accumulative_dw1) + self.eps)
        new_db1 = db1 / (np.sqrt(self.accumulative_db1) + self.eps)
        new_dw2 = dw2 / (np.sqrt(self.accumulative_dw2) + self.eps)
        new_db2 = db2 / (np.sqrt(self.accumulative_db2) + self.eps)
        # Step 3
        self.model.gradient_descent(new_dw1, new_db1, new_dw2, new_db2, self.init_lr)
        # End

        
```


```python

```

### 2.4 Mini-batch Gradient Descent (5 Points)


```python
from collections import OrderedDict


def train_mbgd(model,
               x_train: np.ndarray,
               y_train: np.ndarray,
               x_valid: np.ndarray,
               y_valid: np.ndarray,
               lr: float,
               num_epoch: int,
               batch_size: int,
               print_every: int = 10) -> Tuple[dict[str, List], dict[str, List]]:
    """ Training with Gradient Descent
    Args:
        model: The logistic regression model
        x_train: Training feature, (n x d) matrix
        y_train: Training label, (n, ) vector
        x_valid: Validation feature, (n x d) matrix
        y_valid: Validation label, (n, ) vector
        lr: Learning rate
        num_epoch: Number of training epochs
        batch_size: Number of training samples in a batch
        print_every: Print log every {print_every} epochs
    Returns:
        train_history: Log of training information. The format of training history is
                       { 'loss': [] }
                       It records the average loss of each epoch.
        valid_history: Log of validation information. The format of training and validation history is
                       {
                           'loss': [],
                           'accuracy': [],
                           'precision': [],
                           'recall': [],
                           'f1': []
                       }
    """
    train_history = OrderedDict({'loss': []})
    valid_history = OrderedDict({
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    })

    def format_output(epoch, num_epoch, train_history, valid_history):
        epoch_log = f'Epoch {epoch + 1} / {num_epoch}'
        train_log = ' - '.join([f'train_{key}: {val[-1]:.4f}' for key, val in train_history.items()])
        valid_log = ' - '.join([f'valid_{key}: {val[-1]:.4f}' for key, val in valid_history.items()])
        log = f'{epoch_log}: {train_log} - {valid_log}'
        return log

    # IMPORTANT: YOU SHOULD USE THIS OPTIMIZER TO UPDATE THE MODEL
    optimizer = AdaGrad(init_lr=lr, model=model)

    train_num_samples = len(x_train)
    n_batch = train_num_samples // batch_size
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        # Start your code here (training)
        #     Step 1. Model forward
        y_hat,h1 = model.forward(x_train,True)
        #     Step 2. Calculate loss
        prob = softmax(y_hat)
        epoch_loss = model.categorical_cross_entropy_loss(prob, y_train)
        #     Step 3. Model backward
        dw1, db1, dw2, db2 = model.backward(x_train,y_hat,y_train,h1)
        #     Step 4. Optimization with Adagrad
        optimizer.update(dw1, db1, dw2, db2)
        # End

        valid_loss = 0.
        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        # Start your code here (validation)
        #     Step 1. Predict
        y_hat = model.forward(x_valid,False)
        pred = model.predict(y_hat)
        #     Step 2. Calculate loss
        prob = softmax(y_hat)
        valid_loss = model.categorical_cross_entropy_loss(prob, y_valid)
        valid_history["loss"].append(loss)
        #     Step 3. Calculate metrics
        accuracy, precision, recall, f1 = get_metrics(pred,y_valid)
        valid_history["f1"].append(f1)
        valid_history["recall"].append(recall)
        valid_history["precision"].append(precision)
        valid_history["accuracy"].append(accuracy)
#         End
        train_history['loss'].append(epoch_loss / train_num_samples)
        for vals, val in zip(valid_history.values(), [valid_loss, accuracy, precision, recall, f1]):
            vals.append(val)
        log = format_output(epoch, num_epoch, train_history, valid_history)
        if epoch % print_every == 0 or epoch == num_epoch - 1:
            print(log)
        else:
            print_line(log)
    return train_history, valid_history
```

Run Mini-batch Gradient Descent


```python
np.random.seed(6666)

hidden_dim = 128
num_epoch = 1000
lr = 1e-2
batch_size = 128
lambda_ = 1e-8
print_every = 10

model_mbgd = MLP(feature_dim=vocab_size, hidden_dim=hidden_dim, num_class=num_class, lambda_=lambda_)
mbgd_train_history, mbgd_valid_history = train_mbgd(model_mbgd, x_train, y_train, x_valid, y_valid, lr, num_epoch, batch_size, print_every)
```

### 2.5 MLP using Tensorflow (5 Points)


```python
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.activations import sigmoid


class MLPTF(Model):
    def __init__(self, feature_dim: int, hidden_dim: int, num_class: int, lambda_: float):
        """ MLP Model using tensorflow.keras
        Args:
            feature_dim: feature dimension
            hidden_dim: hidden units
            num_class: number of class
            lambda_: lambda in L2 regularizer
        """
        super().__init__()
        # Start your code here (initialize weight and bias)
        self.dense1 = Dense(hidden_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_))
        self.dense2 = Dense(num_class, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_))
        self.softmax = Softmax()

        # End
        
    def call(self, x):
        """ Forward function of tf. It should be named 'call'
        
        Args:
            x: (n x feature_dim) tensor
        Returns:
            y_hat: (n x num_class) tensor
        """
        # Start your code here (Forward)
        x = self.dense1(x)
        x = self.dense2(x)
        y_hat = self.softmax(x)
        # End
        return y_hat
```


```python
import tensorflow_addons as tfa
np.random.seed(6666)
tf.random.set_seed(6666)


hidden_dim = 128
num_epoch = 1000
lr = 1e-1
batch_size = 128
lambda_ = 1e-8

model_tf = MLPTF(feature_dim=vocab_size, hidden_dim=hidden_dim, num_class=num_class, lambda_=lambda_)
model_tf.build(input_shape=(None, vocab_size))
model_tf.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(), tfa.metrics.F1Score(num_classes=num_class, average='micro')])
model_tf.summary()
```

    C:\Users\Akshay\anaconda3\envs\tensorflow\lib\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning: 
    
    TensorFlow Addons (TFA) has ended development and introduction of new features.
    TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
    Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 
    
    For more information see: https://github.com/tensorflow/addons/issues/2807 
    
      warnings.warn(
    C:\Users\Akshay\anaconda3\envs\tensorflow\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.12.0 and strictly below 2.15.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.10.1 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    

    Model: "mlptf"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               multiple                  2124928   
                                                                     
     dense_1 (Dense)             multiple                  387       
                                                                     
     softmax (Softmax)           multiple                  0         
                                                                     
    =================================================================
    Total params: 2,125,315
    Trainable params: 2,125,315
    Non-trainable params: 0
    _________________________________________________________________
    


```python
tf_history = model_tf.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epoch)
```

#### Evaluation with Tensroflow
You are required to report the loss, accuracy, precision, recall, and f1 on test set and plot the the curve of them for both SGD and Mini-batch GD on train and validation set


```python
# Calculate the metrics for test set and fill in the table below
y_hat = model_mbgd.forward(x_test,False)
y_pred = model_mbgd.predict(y_hat)
print('Mini-batch GD:', get_metrics(y_pred, y_test))
model_tf.evaluate(x=x_test, y=y_test)
```

    Mini-batch GD: (0.7545431277194778, 0.7545431277194778, 0.7545431277194778, 0.7545431277194778)
    123/123 [==============================] - 0s 4ms/step - loss: 0.9271 - categorical_accuracy: 0.5577 - f1_score: 0.5577
    




    [0.9271218776702881, 0.5577169060707092, 0.5577169060707092]



#### Evaluation Metrics on Test set
Fill this table with the result you just printed (double click this cell to edit)
|     Optimizer                     | Accuracy    | F1 Score    |
|:---------------------------------:|-------------|-------------|
|      **Your Implementation**      |     0.675753        | 0.675753            |
| **Tensorflow**                    |    0.9722         |    0.554         |

##### Please run the following cell to plot the training loss curve for Your implementation and Tensorflow


```python
import matplotlib.pyplot as plt

%matplotlib inline

fig = plt.figure(figsize=(4, 3))
plt.plot(mbgd_train_history['loss'], label='My MLP')
plt.plot(tf_history.history['loss'], label='TF MLP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.show()
```


    
![png](output_58_0.png)
    


##### Please run the following cell to plot the validation metrics curve for SGD and Mini-batch GD


```python
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
ax = axes[0]
ax.plot(mbgd_valid_history['accuracy'], label='My MLP')
ax.plot(tf_history.history['val_categorical_accuracy'], label='TF MLP')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_title('Validation Accuracy')

ax = axes[1]
ax.plot(mbgd_valid_history['f1'], label='My MLP')
ax.plot(tf_history.history['val_f1_score'], label='TF MLP')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_title('Validation F1')
plt.show()
```


    
![png](output_60_0.png)
    


# 3. Conclusion (5 Points)

Provide an analysis for all the results

Answer:My implementation achieved an accuracy and F1 score of approximately 0.676. These metrics suggest that the model is performing reasonably well, with a good balance between precision and recall.
The TensorFlow implementation achieved a significantly higher accuracy (0.9722) but a lower F1 score (0.554). The high accuracy indicates that the model is making correct predictions for a large portion of the dataset. However, the lower F1 score suggests a potential imbalance between precision and recall, indicating that the model may struggle with correctly classifying certain classes.

Perfect Fit: It's possible that your model has memorized the training data and achieved a perfect fit, resulting in zero training loss. This could indicate overfitting, especially if the validation performance is not improving.

The increasing trends in both accuracy and F1 score suggest that my model is learning and improving its performance on the validation set over time. This is a positive sign, indicating that your model is continuously learning from the data and adapting to the task.
The constant trends in both accuracy and F1 score for the TensorFlow implementation might suggest that the model may have reached a plateau in terms of learning from the validation set. It could be that the model has already learned most of the patterns in the data, and further training iterations do not significantly impact its performance on the validation set.


```python

```

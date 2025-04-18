# image-caption-generator-using-cnn-lstm
This project implements an image captioning model using the Flickr8k dataset. The model combines Convolutional Neural Networks (CNN) for image feature extraction and Long Short-Term Memory (LSTM) networks for generating captions based on the extracted features.

Table of Contents
Introduction
Requirements
Dataset
Installation
Usage
Model Training
Evaluation
Results
Saving and Loading the Model
License
Introduction
The goal of this project is to generate descriptive captions for images using a combination of deep learning techniques. The model is trained on the Flickr8k dataset, which contains 8,000 images and their corresponding captions.

Requirements
To run this project, you will need the following libraries:

Python 3.x
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Pillow
scikit-learn
tqdm
NLTK
You can install the required libraries using pip:

pip install tensorflow keras numpy pandas matplotlib pillow scikit-learn tqdm nltk

Dataset
The dataset used in this project is the Flickr8k dataset, which can be downloaded from here. The dataset consists of images and their corresponding captions.
Installation
Clone this repository or download the code files.
Place the Flickr8k dataset in the specified directory structure.
Usage
Load the necessary libraries and set the base directory for the dataset.
Run the provided code to preprocess the images and captions.
Train the model using the training data.
Generate captions for new images using the trained model.
Example
To generate a caption for a specific image, use the following function:
generate_caption("102351840_323e3de834.jpg")
Model Training
The model is trained using the following parameters:

Epochs: 12
Batch Size: 32
The training process involves generating image features using the VGG16 model and training the LSTM model to predict captions based on these features.

Evaluation
The model's performance can be evaluated using BLEU scores. The BLEU score measures the similarity between the predicted captions and the actual captions.

Example BLEU Score Calculation
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

Results
After training, the model can generate captions for images. The predicted captions can be visualized alongside the images.

Saving and Loading the Model
The trained model and tokenizer can be saved for future use:
model.save('model.h5')
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

To load the model and tokenizer later, use:
model = load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

License
This project is licensed under the MIT License. See the LICENSE file for more details.


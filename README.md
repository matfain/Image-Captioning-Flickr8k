### Image Captioning project on the Flickr8K dataset
The project is based on this amazing tutorial with slight modifications: https://www.youtube.com/watch?v=y2BaTt1fxJU  

Image captioning is the task of predicting a caption for a given image, which involves using computer vision & NLP methods.  
In this project I used a common encoder-decoder architecture to combine a base CNN model for image feature extraction, with an LSTM based RNN to generate the captions.

Project overview & workflow:
* Data preprocessing- custom dataset & dataloader, tokenization & vocabulary construction
* Model selection - Resnet50 for the encoder CNN, LSTM RNN for the decoder
* Model training - hyperparameters searching and final training
* Inference on the Test set

![image](https://github.com/matfain/Image-Captioning-Flickr8k/assets/132890076/49b03688-9cca-451c-8deb-b4778a51553a)
![image](https://github.com/matfain/Image-Captioning-Flickr8k/assets/132890076/c7674c71-e970-4a82-ab80-2c1fd035ef51)

# Clothing Predictor App

The output of this project is a flask app that takes an Instagram handle or image URL, looks for images with one person in them, and then makes a prediction on what the person is wearing. Right now the model is trained on 13 different clothing categories.

The purpose of this project was to take a first step towards building a robust image classification tool for fashion. It also gave me an opportunity to explore Keras and learn more about convolutional neural networks.

### Approach
Using Keras, I fine-tuned Google's inception V3 in two different ways to build the app.  I trained a "filter model" to identify images with only one person in them. Right now I am only making one prediction per photo, so I wanted to make sure my app wasn't making a clothing prediction on inanimate objects or photos of a group of people.  
  
The "clothing predictor model" is used to make the actual prediction on clothing type. 

Both models converged to accuracy levels of 85-90% on validation image sets.

### Presentation
https://prezi.com/qco8gepbmsoa

### Data
[Fashion 10000](https://www.researchgate.net/publication/262254329_Fashion_10000_An_enriched_social_image_dataset_for_fashion_and_clothing "Paper") - This dataset included roughly 32,000 Flickr photos with mechanical turk annotations. One of the questions asked of the annotators was "How many people are in this photo?". The answer to this question allowed me to split the dataset and train on single-subject images.

Flickr/Google Image Search - The clothing predictor model was trained with roughly one thousand photos per category, hand picked from query results on Google and Flickr. I tried to select photos with rich and diverese backdrops, as this helped the model generalize better when predicting photos of people in natural settings. 

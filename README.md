## At Long Last -- a Hotdog Detector!

Tensorflow 2.0 version of the ol' hotdog detector.

Sample hotdog and non-hotdog food images are zipped up under src/resources/images. If you come across any more hotdogs pics, please pass along to your humble servant.

Once you've split the pics into training and validation sets (there's a utility included to help with that), src/cnn/train.py will train a CNN, save it to disk, and display charts of training vs. validation loss and accuracy.

If you have an Android phone you should be able to use your saved model with the [Android app](https://github.com/futonchild/hot_or_notdog2) available from my GitHub.


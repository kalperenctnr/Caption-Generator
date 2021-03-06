# Caption-Generator
## Caption Generation from Flickr Images
The main task of this project is to create a caption generator network. Initially, this task can be decomposed into two categories, preprocess and process parts. In the preprocess part, we created dictionaries that provide mappings from images to captions and captions to images. Moreover, we have decoded and resized images to fit them to the encoding process. This is recommended as original image sizes differ and to have more robust features, it is plausible to have the inputs as the same size. In the process part, first task was to extract features from images via a CNN network (encoder) and use them along with caption data to generate captions in a RNN network (decoder). The idea is that the set by which CNN is pretrained is going to have a similar data distribution with our data. There are two basic approaches to these image generator networks. One of which is “Merging Architecture” and the other is “Injecting Architecture” . In Merging Architecture, the image is not introduced in the RNN network. 
Therefore, the image and the word data are encoded separately. They are introduced together in feed-forward network which creates multimodal layer architecture. In Injecting Architecture, even though there are variants of injecting architecture according to the step at which image data is injected, the general structure is that word and image data is fed into the RNN network together.
![](description_images/github.png)

Figure 1: Injecting&Merge Architecture

Moreover, in order to feed image data to the RNN network, a CNN network is used to extract features from images then these images are fed into the RNN network. We have passed all images from the CNN network and obtained the useful features at the layer before FC layers of the CNN. Since, we make use of the Injecting Architecture; these features are concatenated to embedding outputs of the corresponding words. At the RNN part, words are fed into the system one by one since they hit the last element the learning process is illustrated in the following figure:

![](description_images/steps.png)

Figure 2: Training Steps

In addition, there is an attention mechanism whose task is to search for a set of positions in a source where the most relevant information is concentrated while the model is trying to predict the next word. Hence, the less relevant parts of the input vector are given less weights which affect the state transitions in the decoder part of the model.

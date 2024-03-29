# NN_Project
## Description:
This repo is for the Semester Project for Neural Networks. The group members are Seth Pickford(spickfor) and Alex Castillo (acastil5)


## Part 1:
We aren't exactly sure which dataset we are going to use.  We have looked over the suggested datasets and can see that either FRGC's dataset or 
BioID's dataset could be viable to use in this project.  We are leaning towards BioID right now because it doesn't have quite as many images,
 which could allow quicker training and it also focuses on frontal facing images which will most likely be easier to train a model on.
However, we aren't exactly sure how much time it will take to train so we could end up switching because more datapoints means that training 
can be more effective.  Along these same lines, if we don't think that we have enough data, we can lean on the creation of synthetic images to 
give us more information to train with.  With the data, we will have to do data preprocessing.  This will most likely involve normalizing the 
images and resizing them if needed so that they have the same dimensions.  We aren't exactly sure if we will use a third party software to do 
the facial detection itself.  We think it could be interesting to come up with this on our own, but also don't want to spend a lot of time 
implementing this portion of our project.  After our facial detection, the faces themselves will have to be cropped and properly aligned so 
that the inputs into our NN are properly standardized.  This will be extremely important in making sure that slight variations in our input 
data that don't go to our main features aren't throwing off our outputs and what the model is actually learning. For the model's facial 
detection, we don't know a whole lot on how to implement it yet, but based on suggestions, the network will output vector of features that can 
be compared using different metrics in order to determine how similar they are.  The main features that will be looked at will be the eyes, 
nose and mouth.  To figure out similarities we can compare certain things such as distances between these certain main facial points.  We 
thought about possibly using eye color, but thought that this could vary too much based on lighting and other environmental factors.

Some things from class that could help us with this project are the things we have recently learned with the identification of numbers.  We can 
use kernels to look at our image and possibly experiment with different sized kernels to find one that is effective at looking at human faces.  
I think that the google colab for the number recognition could be a very helpful starting point that we can reference while learning how to go 
about creating our project.

For this part we worked together to brainstorm ideas and how we could possibly go about creating the project.  We also used ChatGPT to help us 
brainstorm and think of an outline of how to go about creating the project.


## Part 2

Source: https://www.bioid.com/face-database/
There are a couple of key things that we need to focus on for the training versus the validation subsets.  First, we need to make sure that
both subsets contain images that have variability in lighting and background.  We want the photos of our faces to be varied between the 
subsets so that the model can better generalize all these different conditions, rather than learning to do only one specific looking type
of black and white photo.  Next we wnat a variety of poses and facial expressions.  We don't just want one type of facial expression to
lead to biases within our model.  We also want to make sure that we are representing all of our subjects equally.  We don't want more photos
of one person to show up than others in our different subsets because, say it happens in our training set, it could lead to bias in our model
where it becomes overly tuned to one specific group or one specific person and then struggles to generalize other groups.
The BioID dataset contains 1521 gray scale images which represent 23 different people.  This means that each person has about 66 samples, with
some having more and some having less.  This needs to be looked into further to make sure that these differences aren't large enough to cause
any issues in training our model.  Each photo is 384x286 pixels and is rendered in black and white.  Additionally, each photo has had the eye 
positions of the person set manually.  Additionally, the dataset contains files other than the photos.  These files contain the x and y
coordinates of the person's eyes, which will be extremely useful for calculating facial distances and positioning.
When trying to access the FRGC database, we got a message saying "We're sorry, we were unable to locate the site /public-cvrl/data-sets."
Additionally, the professor suggested using StyleGAN for additional testing; however, we aren't sure how this could be done because StyleGAN
produces a new face every time from what we can tell, which means that it wouldn't be useful for facial recognition training and testing.  It
 would have to be the same image in both trainging and testing, which means that it would create a model that wouldn't be able to generalize new
faces.
Alex has found a database called Casia-Webface that contains almost 500,000 images of 10,500 different people. (Source: https://www.kaggle.com/datasets/debarghamitraroy/casia-webface?resource=download).  The differences between training and validation should be the same as mentioned
above.  The images in this dataset are varied in their resolution and quality, which could lead to some issues and will mean that those that
are used will have to be preprocesses to make them as similar as possible.  Additionally, they vary from BioID in that they are color photos
not black and white.
We didn't have time this week to talk with the professor, because our concept design response came out mid week and we both had busy weeks.
We plan to talk to the professor some time this week to figure out his thoughts on this new dataset.
Additionally, we know that Note 1 says not to present data that we "plan" to acquire, but we have this dataset downloaded and can use it,
we just aren't sure how viable it could be.


Each of us has looked at the datasets and analyzed their photos to figure out how we can use them and to figure out what needs to 
be different between our subsets to try and get the most out of our data.
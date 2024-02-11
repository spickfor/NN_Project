# NN_Project
Description:
This repo is for the Semester Project for Neural Networks. The group members are Seth Pickford(spickfor) and Alex Castillo (acastil5)


Part 1:
    We aren't exactly sure which dataset we are going to use.  We have looked over the suggested datasets and can see that either FRGC's dataset or BioID's dataset could be viable
    to use in this project.  We are leaning towards BioID right now because it doesn't have quite as many images, which could allow quicker training and it also focuses on frontal 
    facing images which will most likely be easier to train a model on.  However, we aren't exactly sure how much time it will take to train so we could end up switching because more datapoints means that training can be more effective.  Along these same lines, if we don't think that we have enough data, we can lean on the creation of synthetic images to give us more information to train with. 
    With the data, we will have to do data preprocessing.  THis will most likely involve normalizing the images and resizing them if needed so that they have the same dimensions.  We aren't exactly sure if we will use a third party software to do the facial detection itself.  We think it could be interesting to come up with this on our own, but also don't want
    to spend a lot of time implementing this portion of our project.  After our facial detection, the faces themselves will have to be cropped and properly aligned so that the inputs
    into our NN are properly standardized.  This will be extremely important in making sure that slight variations in our input data that don't go to our main features aren't throwing off our outputs and what the model is actually learning.
    For the model's facial detection, we don't know a whole lot on how to implement it yet, but based on suggestions, the network will output vector of features that can be compared using different metrics in order to determine how similar they are.  The main features that will be looked at will be the eyes, nose and mouth.  To figure out similarities we can compare certain things such as distances between these certain main facial points.  We thought about possibly using eye color, but thought that this could vary too much based on lighting and other environmental factors.
    Some things from class that could help us with this project are the things we have recently learned with the identification of numbers.  We can use kernels to look at our image and possibly experiment with different sized kernels to find one that is effective at looking at human faces.  I think that the google colab for the number recognition could be a very helpful starting point that we can reference while learning how to go about creating our project.

For this part we worked together to brainstorm ideas and how we could possibly go about creating the project.  We also used ChatGPT to help us brainstorm and think of an outline of how to
go about creating the project.
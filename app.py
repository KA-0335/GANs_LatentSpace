import os
import json
import pickle
import random
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot
from pathlib import Path
import plotly.express as px
from matplotlib import pyplot
from sklearn.manifold import TSNE
from scipy.spatial import distance
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from model import latent_points, model
from flask import Flask, render_template, request

#loading pickle file

path = 'FLSK/'
path_data = '/FLSK/cc'
path_static = 'FLSK/static'

images, pca_features, pca = pickle.load(open(path+'feature_flowers.p', 'rb'))

for img, f in list(zip(images, pca_features))[0:5]:
    print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... "%(img, f[0], f[1], f[2], f[3]))

num_images_to_plot = 25

if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]
X = np.array(pca_features)
#t-SNE parameters
tsne = TSNE(n_components=2, learning_rate=100, perplexity=22).fit_transform(X)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

#dimensions of the graph
width = 3000
height = 3000
max_dim = 500

#plotting the figure
full_image = Image.new('RGBA', (width, height))
for img, x, y in zip(images, tx, ty):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

matplotlib.pyplot.figure(figsize = (20,16))
imshow(full_image)
#saving the figure
plt.savefig(path_static+'//plot.png',bbox_inches='tight')

#%%

# Define the Flask application
app = Flask(__name__)

#interpolating between points based on user input
def interpolate_point(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = []
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)

#generating frames or images for the gif
def image_transition_2(images, n, i, j):
    names = []
    for x in range(n):
        pyplot.axis('off')
        pyplot.imshow(images[x, :, :])
        name = "img"+str(x)+".png"
        #saving figure to png
        pyplot.savefig(name,bbox_inches='tight')
        names.append(name)
    return names

#Create GIFs
def transition_creation_2(image_name,i,j):
    #going through all the images from image_transition_2
    with imageio.get_writer(('morphing.gif'), mode='I') as writer:
        for filename in image_name:
            image = imageio.imread(filename)
            writer.append_data(image)

#avoiding caching
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#initialise the FLASK app
#Setting the route
@app.route("/")
def express():  
    print("")
    #render template returns the first webpage to be displayed
    return render_template("graph.html")

#The function gets executed after the render template, using the post method 
@app.route('/predict', methods=['POST'])
def graph():
    #taking user input
    data1 = request.form['p1']
    data2 = request.form['p2']
    #Changing directory for images to be displayed
    os.chdir(path_static)
    #function calling for the GAN model to interpolate and featrue arithmetic
    generate_images(data1, data2)
    os.chdir(path)
    #render template returns the second webpage to be displayed
    return render_template('after.html')

#interpolating between points based on user input
def interpolate_point(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = []
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)

#generating frames or images for the gif
def image_transition_2(images, n, i, j):
    names = []
    for x in range(n):
        pyplot.axis('off')
        pyplot.imshow(images[x, :, :])
        name = "img"+str(x)+".png"
        #saving figure to png
        pyplot.savefig(name,bbox_inches='tight')
        names.append(name)
    return names

#Create GIFs
def transition_creation_2(image_name,i,j):
    #going thorugh all the images from image_transition_2
    with imageio.get_writer(('morphing.gif'), mode='I') as writer:
        for filename in image_name:
            image = imageio.imread(filename)
            writer.append_data(image)

def average_points(points, ix):
    # retrieve required vectors corresponding to the selected images
    vectors = points[ix]
    # average the vectors
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector

#Function to take user input from the webpage and implement interpolation
def generate_images(i,j):
    #user inputs  
    i = int(i)
    j = int(j)
    #getting them in lists
    flower_1 = [i]
    flower_2 = [j]

    # average vectors for each class
    feature1 = average_points(latent_points, flower_1)
    feature2 = average_points(latent_points, flower_2)
    # Vector arithmetic....
    result_vector = np.expand_dims((feature1 + feature2), 0)
    #passing the resultant vector through the model for prediction
    result_image = model.predict(result_vector)

    #scale pixel values for plotting
    result_image = (result_image + 1) / 2.0
    plt.imshow(result_image[0])
    #saving the figure
    plt.savefig(path_static+'//add.png',bbox_inches='tight')
  
    if i != j:
        tr = interpolate_point(latent_points[i], latent_points[j], 50)
        interpolated = tr * 1
        X = model.predict(interpolated)
        X = (X + 1) / 2.0
        # names = image_transition(X, len(interpolated), i,j)
        # transition_creation(names)
        zz = image_transition_2(X, len(interpolated), i, j)
        transition_creation_2(zz,i,j)
    return None    

#main function to run FLASK APP
if __name__ == '__main__':
    app.debug = False
    app.run()

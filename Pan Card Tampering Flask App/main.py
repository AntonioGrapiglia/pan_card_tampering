# Interesting boiler-plate code to scrap data 
# Basic imports used in the id tampering project
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image 
import requests

# Adding both the original and tempared image to variables
original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg', stream=True).raw)
tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png', stream=True).raw)

# The file format of the source file
print('Original image format :', original.format)
print('Tampered image format :', tampered.format)

# The size of the images in pixels given by tuples of width and height
print('Original image size :', original.size)
print('Tampered image size :', tampered.size)

# Resizing image and saving it to folders(uncomment save in case of new images)
original = original.resize((250,160))
#original.save('pan_card_tampering/image/original.png') 
tampered = tampered.resize((250, 160))
#tampered.save('pan_card_tampering/image/tampered.png')

# Simply here to load the images into cv2 to be able to run cv2 fucntions into them
original = cv2.imread('pan_card_tampering/image/original.png')
tampered = cv2.imread('pan_card_tampering/image/tampered.png')

# Now converting the images to b&w since the important features are cornes and such
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

# Now the Structural Similarity Index(SSIM) is computed
(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff*255).astype('uint8')
print('SSIM: {}'.format(score))

# Getting countours
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Creating bounding rectangle over image 
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Visualizing images
original_contour = Image.fromarray(original)
original_contour.save('outputs/original_contour.png')
original_contour.show()

tampered_contour = Image.fromarray(tampered)
tampered_contour.save('outputs/tampered_contour.png')
tampered_contour.show()

Diff = Image.fromarray(diff)
Diff.save('outputs/Diff.png')
Diff.show()

threshimg = Image.fromarray(thresh)
threshimg.save('outputs/thresh.png')
threshimg.show()


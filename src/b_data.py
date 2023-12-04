import zipfile

def prepare_data(bconfig):
  with zipfile.ZipFile(bconfig['brazil_data'], "r") as z:
    z.extractall("\data")
  image1 = []
  image2 = []
  image3 = []
  image4 = []
  image5 = []
  folder_dir = '/data/brazilian_coffee_scenes/fold1'
  for images in os.listdir(folder_dir):
    if (images.endswith(".jpg")):
        image1.append(images)
  folder_dir = '/data/brazilian_coffee_scenes/fold2'
  for images2 in os.listdir(folder_dir):
    if (images2.endswith(".jpg")):
        image2.append(images2)
  folder_dir = '/data/brazilian_coffee_scenes/fold3'
  for images3 in os.listdir(folder_dir):
    if (images3.endswith(".jpg")):
        image3.append(images3)
  folder_dir = '/data/brazilian_coffee_scenes/fold4'
  for images4 in os.listdir(folder_dir):
    if (images4.endswith(".jpg")):
        image4.append(images4)
  folder_dir = '/data/brazilian_coffee_scenes/fold5'
  for images5 in os.listdir(folder_dir):
    if (images5.endswith(".jpg")):
        image5.append(images5)
  all_images = image1 + image2 + image3 + image4 + image5

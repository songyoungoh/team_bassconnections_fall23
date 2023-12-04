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
  os.chdir('/data/brazilian_coffee_scenes')
  fold1file = open("fold1.txt")
  fold1content = fold1file.read()
  fold1split = fold1content.split()
  labels1 = []
  count = 0
  fold2file = open("fold2.txt")
  fold2content = fold2file.read()
  fold2split = fold2content.split()
  labels2 = []
  fold3file = open("fold3.txt")
  fold3content = fold3file.read()
  fold3split = fold3content.split()
  labels3 = []
  fold4file = open("fold4.txt")
  fold4content = fold4file.read()
  fold4split = fold4content.split()
  labels4 = []
  fold5file = open("fold5.txt")
  fold5content = fold5file.read()
  fold5split = fold5content.split()
  labels5 = []
  for label in fold1split:
    labels1.append(label.split('.',1))
  for label in fold2split:
    labels2.append(label.split('.',1))
  for label in fold3split:
    labels3.append(label.split('.',1))
  for label in fold4split:
    labels4.append(label.split('.',1))
  for label in fold5split:
    labels5.append(label.split('.',1))
  all_labels = labels1 + labels2 + labels3 + labels4 + labels5
  all_labels = sorted(all_labels, key=lambda x: x[1])
  noncoffee = [], coffee = []
  for pair in all_labels:
    if (pair[0] == "noncoffee"):
      noncoffee.append(pair[1] + ".jpg")
    elif (pair[0] == "coffee"):
      coffee.append(pair[1] + ".jpg")
  collection = "/data/brazilian_coffee_scenes/fold1"
  for i, filename in enumerate(os.listdir(collection)):
  if(filename in coffee):
      os.rename("/data/brazilian_coffee_scenes/fold1/" + filename, "coffee" + filename)
  elif(filename in noncoffee):
      os.rename("/data/brazilian_coffee_scenes/fold1/" + filename, "non" + filename)
  collection = "/data/brazilian_coffee_scenes/fold2"
  for i, filename in enumerate(os.listdir(collection)):
    if(filename in coffee):
      os.rename("/data/brazilian_coffee_scenes/fold2/" + filename, "coffee" + filename)
    elif(filename in noncoffee):
      os.rename("/data/brazilian_coffee_scenes/fold2/" + filename, "non" + filename)
  collection = "/data/brazilian_coffee_scenes/fold3"
  for i, filename in enumerate(os.listdir(collection)):
  if(filename in coffee):
    os.rename("/data/brazilian_coffee_scenes/fold3/" + filename, "coffee" + filename)
  elif(filename in noncoffee):
    os.rename("/data/brazilian_coffee_scenes/fold3/" + filename, "non" + filename)
  collection = "/data/brazilian_coffee_scenes/fold4"
  for i, filename in enumerate(os.listdir(collection)):
    if(filename in coffee):
      os.rename("/data/brazilian_coffee_scenes/fold4/" + filename, "coffee" + filename)
    elif(filename in noncoffee):
      os.rename("/data/brazilian_coffee_scenes/fold4/" + filename, "non" + filename)
  collection = "/data/brazilian_coffee_scenes/fold5"
  for i, filename in enumerate(os.listdir(collection)):
    if(filename in coffee):
      os.rename("/data/brazilian_coffee_scenes/fold5/" + filename, "coffee" + filename)
    elif(filename in noncoffee):
      os.rename("/data/brazilian_coffee_scenes/fold5/" + filename, "non" + filename)
  coffee_dir = os.path.join("/data", "coffee")
  noncoffee_dir = os.path.join("/data", "test")
  os.makedirs(coffee_dir, exist_ok=True)
  os.makedirs(noncoffee_dir, exist_ok=True)
  collection = "/data/brazilian_coffee_scenes"
  for file in enumerate(os.listdir(collection)):
    if "coffee" in file[1]:
      shutil.move(os.path.join(collection, file[1]),os.path.join("/data/Coffee", file[1]))
    elif "non" in file[1]:
      shutil.move(os.path.join(collection, file[1]),os.path.join("/data/NonCoffee", file[1]))
  def train_test_split_folder(base_dir, output_dir, test_size=0.2):
    #Creating train and test folders
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # List classes based on subdirectory names
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for cls in classes:
        # Create output directories for this class
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        # List all jpg images in this class
        images = [f for f in os.listdir(os.path.join(base_dir, cls)) if f.endswith(('.jpg'))]
        # Shuffle the images
        random.shuffle(images)
        # Split point
        split_point = int(len(images) * (1 - test_size))
        # Split the images into train and test sets
        train_images = images[:split_point]
        test_images = images[split_point:]
        # Move the images
        for img in train_images:
            shutil.copy(os.path.join(base_dir, cls, img), os.path.join(train_dir, cls, img))
        for img in test_images:
            shutil.copy(os.path.join(base_dir, cls, img), os.path.join(test_dir, cls, img))
        print(f"Processed {cls} - {len(train_images)} training, {len(test_images)} test")

  train_test_split_folder("/content", "/content", test_size=0.2)

  train_dir = '/content/train'

train_images = []
train_labels = []
for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)

    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = Image.open(image_path)
        image_array = np.array(image)
        #image_array[:,:,0] = np.zeros((64,64))
        #image_array[:,:,1] = np.zeros((64,64))
        #image_array[:,:,2] = np.zeros((64,64))
        train_images.append(image_array)
        train_labels.append(label)

# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_dir = '/content/test'

# Initialize empty lists to store images and labels
test_images = []
test_labels = []

# Iterate through each subfolder
for label in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label)

    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = Image.open(image_path)
        image_array = np.array(image)
        #image_array[:,:,0] = np.zeros((64,64))
        #image_array[:,:,1] = np.zeros((64,64))
        #image_array[:,:,2] = np.zeros((64,64))
        test_images.append(image_array)
        test_labels.append(label)

# Convert the lists to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

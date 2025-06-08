import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import random

#Data augmentation
data_generator = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    shear_range = 0.1,
    fill_mode = 'nearest'
)


TARGET_COUNT = 400


class_path = r"D:\My LPR project\Model_training\new_dataset\train\class_I"
    
#get the current image 
current_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpeg', '.jpg'))]
current_count = len(current_images)
images_needed = TARGET_COUNT - current_count
print(f"\n processing : {class_path}")
print(f"current images: {current_count} need ro generate: {images_needed}")

if images_needed <= 0:
    print("skipping already enough images")
else:
    if current_count == 0:
        print("Error: No source image sfound for augmentation")
    else:
        #shuffling images to ensure balanced augmentation
        random.shuffle(current_images)

        generated = 0
        i = 0
        while generated < images_needed:
            img_name = current_images[i]
            img_path = os.path.join(class_path, img_name)
            try:
                #load and augment image
                img = load_img(img_path)
                img_arr = img_to_array(img)
                img_arr = np.expand_dims(img_arr, axis= 0) #convert to batch format

                #generate one augmented image
                aug_generator = data_generator.flow(
                    img_arr,
                    batch_size = 1,
                    save_to_dir = class_path,
                    save_prefix = 'aug',
                    save_format = 'png'
                )
                next(aug_generator) #generate and save one image
                generated +=1
                print(f"generated {generated}(generated)/{images_needed}(image needed) from {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
            #move to the nex image (circular iteration)
            i = (i + 1) % current_count
        print(f"Successfully generated {generated} augmented images")
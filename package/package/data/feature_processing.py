from package.utils.file_system import get_root_project_path
from package.utils.file_system import read_file
import pandas as pd
import os 
def create_feature_metadata_table():
    """
    Process the dataset to create a feature metadata table.
    This table contains the textual features and their corresponding images.
    """
    pass 

def process_textual_data():
    """
    Process the textual data to create a feature metadata table.
    This table contains the textual features and their corresponding images.
    """
    textual_data = read_file("data/raw/HousesInfo.txt")
    lines = textual_data.split("\n")
    feature_metadata = []
    for n, line in enumerate(lines):
        if line.strip():  # Ignore empty lines
            n_bedroons, n_bathrooms, area, zipcode, price = line.split(" ")
            feature_metadata.append({
                "id_": n+1,
                "n_bedroons": float(n_bedroons),
                "n_bathrooms": float(n_bathrooms),
                "area": float(area),
                "zipcode": float(zipcode),
                "price": float(price)
            })
            
    # Convert the dictionary to a DataFrame or any other structure you prefer
    # For example, using pandas:
    df = pd.DataFrame(feature_metadata)
    print(df.head())

def process_images():
    """
    Process the images to create a feature metadata table.
    This table contains the image features and their corresponding textual data.
    """
    root = get_root_project_path()
    
    print(root.parent.parent / "HOUSES-DATASET"/"Houses Dataset")
    
    images_list = os.listdir(root.parent.parent / "HOUSES-DATASET"/"Houses Dataset")
    images_list = [image for image in images_list if image.endswith(".jpg")]
    image_metadata = []
    for image in images_list:
        image_path = root.parent.parent / "HOUSES-DATASET"/"Houses Dataset"/image
        index = image.split("_")[0]
        image_metadata["id_"] = int(index)  
        image_metadata[image_path.name.split("_")[1].split(".")[0]] = image_path


    print(image_metadata)
        
        
        # Process the image as needed (e.g., load, resize, etc.)
        # For example, using PIL:
        # img = Image.open(image_path)
        # img = img.resize((128, 128))  # Resize to 128x128 pixels
        # img.show()  # Display the image
    

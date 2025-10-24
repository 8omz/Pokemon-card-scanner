import cv2
import tempfile
import os
import datetime

# configuration and constants 
# paths
dataset_csv_path="data/manifests/dataset.csv"
overlay_dir_path="data/annotated/phone/"
annotation_log_path = "data/manifests/annotation_log.csv"
#window settings
window_name = "Annotator - {image_id} ({i}/{N})"
max_window_width = 1600
max_window_height = 1200
#visual settings
corner_colors = (0, 0, 255)  # color for corner points
corner_radius = 5             # radius for corner points
valid_line_color = (0, 255, 0)   # color for valid lines
invalid_line_color = (255, 0, 0) # color for invalid lines
line_thickness = 2           # thickness for lines
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 255, 255)
#key binds 
key_quit = ord('q') # Q
key_reset = ord('r') # R
key_confirm = 13   # Enter
key_skip = 27     # Esc
key_undo = 8       # Backspace  

#validation settings
min_side_length = 20  # minimum length for each side of the quadrilateral
min_area = 1000       # minimum area for the quadrilateral
max_angle_deviation = 15  # maximum deviation from 90 degrees for each angle
max_parallel_deviation = 15 # maximum deviation from parallelism for opposite sides
min_corners_distance = 10  # minimum distance between corners to avoid overlap

# coordinates ordering
# Corner order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
ordered_coords = ["TL", "TR", "BR", "BL"]
# Invariants:
# - All stored corner coordinates are in ORIGINAL pixel space (not scaled).
# - Corners always ordered TL → TR → BR → BL.
# - Manifest is written immediately after each confirmed annotation.

# manifest controller functions
def check_manifest_exists(path):
    try:
        open(path,"r")
    except:
        print("No dataset.csv file could be found/opened. Please review the label_schema.md file for instructions...")
        return False
    
def read_metadata(path):
    fv = open(path,"r")
    photo_list = []
    seen_image_ids = set()

    with fv as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line.startswith("image_id"):
                continue
            line = line.strip("\n")
            split =line.split(",")
            if split[0] in seen_image_ids:
                print(f"Warning: Duplicate image_id {split[0]} found. Skipping duplicate entry.")
                continue
            #debug print(split)
            photo = {
                "image_id":split[0],
                "image_path":split[1],
                "device":split[2],
                "set_code":split[3],
                "set_number":split[4],
                "variant":split[5],
                "angle":split[6],
                "distance_cm":split[7],
                "TL_x":int(split[8]) if split[8] else None,"TL_y":int(split[9]) if split[9] else None,
                "TR_x":int(split[10]) if split[10] else None,"TR_y":int(split[11]) if split[11] else None,
                "BR_x":int(split[12]) if split[12] else None,"BR_y":int(split[13]) if split[13] else None,
                "BL_x":int(split[14]) if split[14] else None,"BL_y":int(split[15]) if split[15] else None
            }
            seen_image_ids.add(photo["image_id"])
            photo_list.append(photo)
    return photo_list

def load_manifest(path):
    if check_manifest_exists(path) is not False:
        print("Valid file found...")
    else:
        return 
    if read_metadata(path) is not None:
        print("Metadata read successfully...")
    print((f"A total of {len(read_metadata(path))} valid photos were found..."))

    return read_metadata(path)
    
    
def find_unannotated_rows(table):
    unannotated_list = []
    for i in range(len(table)):
        photo = table[i]
        if photo["TL_x"] == None or photo["TL_y"] == None or photo["TR_x"] == None or photo["TR_y"] == None or photo["BR_x"] == None or photo["BR_y"] == None or photo["BL_x"] == None or photo["BL_y"] == None:
            unannotated_list.append(i)
    return unannotated_list
def get_row(table,idx):
    return table[idx]
    
photo_table = load_manifest(dataset_csv_path)
# expects four_corners to be a list of tuples [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
def write_corners(table,idx:int,four_corners:list):
    photo = table[idx]
    photo["TL_x"] = four_corners[0][0]
    photo["TL_y"] = four_corners[0][1]
    photo["TR_x"] = four_corners[1][0]
    photo["TR_y"] = four_corners[1][1]
    photo["BR_x"] = four_corners[2][0]
    photo["BR_y"] = four_corners[2][1]
    photo["BL_x"] = four_corners[3][0]
    photo["BL_y"] = four_corners[3][1]
    
    
    
write_corners(photo_table,0,[(None,None),(None,None),(None,None),(None,None)])
def save_manifest(table,path):
    temp = tempfile.NamedTemporaryFile(dir="data/manifests/", delete=False)
    count = 0
    temp.close()
    with open(temp.name,"w") as file:
        file.write("image_id,image_path,device,set_code,set_number,variant,angle,distance_cm,corner_tl_x,corner_tl_y,corner_tr_x,corner_tr_y,corner_br_x,corner_br_y,corner_bl_x,corner_bl_y\n")
        for photo in table:
            string = f'{photo["image_id"]},{photo["image_path"]},phone,{photo["set_code"]},{photo["set_number"]},{photo["variant"]},{photo["angle"]},{photo["distance_cm"]},{photo["TL_x"] if photo["TL_x"] is not None else ""},{photo["TL_y"] if photo["TL_y"] is not None else ""},{photo["TR_x"] if photo["TR_x"] is not None else ""},{photo["TR_y"] if photo["TR_y"] is not None else ""},{photo["BR_x"] if photo["BR_x"] is not None else ""},{photo["BR_y"] if photo["BR_y"] is not None else ""},{photo["BL_x"] if photo["BL_x"] is not None else ""},{photo["BL_y"] if photo["BL_y"] is not None else ""}\n'
            file.write(string)
            count += 1
            
    print(f"Wrote {count} entries to  temp manifest...")
    try:
        os.rename(temp.name,path)
        print(f"Renamed new manifest to dataset.csv.tmp...")
    except:
        os.remove(path)
        os.rename(temp.name,path)
        print("Replaced old manifest with new manifest...")
        pass
    print(f"Manifest saved successfully — {count} records written to {path}")

def append_log(image_id,status,reason=None):
    file_exists = os.path.exists(annotation_log_path)
    need_header = False
    if not file_exists or (file_exists and os.path.getsize(annotation_log_path)) == 0:
        need_header = True
    if need_header:
          with open(annotation_log_path,"a") as file:
              file.write("TIMESTAMP,image_id,status,reason\n")
    with open(annotation_log_path,"a") as file:
        
        current_time = datetime.datetime.now().isoformat()
        if status == "ANNOTATED":
            file.write(f"{current_time},{image_id},{status}\n")
        else:
            file.write(f"{current_time},{image_id},{status},{reason}\n")
    


# Image I/O & Scaling
def read_image(image_id,image_path):

    try:
        cv2.imread(image_path)
    except Exception as e:
        append_log(image_id,"READ_IMAGE_ERROR",reason=f"Error during reading: {e}")
        return None
    if cv2.imread(image_path) is None:
        append_log(image_id, "READ_IMAGE_ERROR", reason=f"Image exists but could not be decoded by OpenCV.")
        return None
    else:
        append_log(image_id,"IMAGE_READ",reason=f"Image Read sucessfully")

        
read_image(photo_table[0]["image_id"],photo_table[0]["image_path"])
read_image(photo_table[1]["image_id"],"hi")
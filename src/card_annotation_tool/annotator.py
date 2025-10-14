import cv2
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
print(find_unannotated_rows(photo_table))
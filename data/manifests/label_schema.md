# Pokémon Card Scanner — Label Schema

## Document Overview
This file defines the data schema for Pokémon Card Scanner annotations.  
Each entry in `dataset.csv` corresponds to one physical Pokémon card image.  
The goal is to record both **metadata** (set, number, variant, etc.) and **geometric ground-truth coordinates** (the four card corners in pixel units).

---

### File Reference
- **File:** `data/manifests/dataset.csv`  
- **Format:** CSV (comma-separated)  
- **Rows:** One per image  

---

## Definitions

Each column in `dataset.csv` has a specific purpose and definition.

### Metadata Columns

| Column | Description |
| ------ | ------------ |
| `image_id` | Unique identifier following a standardized naming convention for each image. |
| `image_path` | Relative path to the image within the folder structure. |
| `device` | Device the image was captured on (e.g., phone, webcam). |
| `set_code` | Pokémon card set code (e.g., `BLK` = Black Bolt, `SSH` = Sword & Shield). |
| `set_number` | Card number within the set (e.g., Serperior’s set number is 156). |
| `variant` | Card variation type: normal, reverse holographic, regular holographic, EX, GX, etc. |
| `angle` | Capture angle category (e.g., `topdown`, `tiltleft`, `yawtop`, etc.). |
| `distance_cm` | Distance between camera lens and card surface in centimeters. |

---

### Annotation Columns

The coordinates recorded during manual corner annotation.

| Column | Description |
| ------- | ------------ |
| `corner_tl_x` | X coordinate of the top-left corner. |
| `corner_tl_y` | Y coordinate of the top-left corner. |
| `corner_tr_x` | X coordinate of the top-right corner. |
| `corner_tr_y` | Y coordinate of the top-right corner. |
| `corner_br_x` | X coordinate of the bottom-right corner. |
| `corner_br_y` | Y coordinate of the bottom-right corner. |
| `corner_bl_x` | X coordinate of the bottom-left corner. |
| `corner_bl_y` | Y coordinate of the bottom-left corner. |

> **Coordinate System:** (0, 0) is the top-left corner of the image.  
> **Units:** Pixels.  
> **Corner Order:** Always recorded in clockwise order — TL → TR → BR → BL.

---

## Annotation Rules

A valid annotation must satisfy all of the following conditions:

- All 8 coordinate fields (X and Y for TL, TR, BR, BL) must be populated.  
- No two corners may be within ~10 pixels of each other.  
- The four points should form an approximate rectangle (within perspective tolerance).  
- Click order must always follow **TL → TR → BR → BL** (top-left clockwise).  
- Coordinate precision tolerance: ±3 pixels from the visible card boundary.  

---

## Example Entry

| image_id | image_path | device | set_code | set_number | variant | angle | distance_cm | corner_tl_x | corner_tl_y | corner_tr_x | corner_tr_y | corner_br_x | corner_br_y | corner_bl_x | corner_bl_y |
|-----------|-------------|---------|-----------|-------------|----------|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| BLK_046_ex__phone_tiltleft_04 | data/raw/phone/BLK_046_ex__phone_tiltleft_04.jpg | phone | BLK | 046 | ex | tiltleft | 7 | 123 | 88 | 422 | 92 | 430 | 605 | 120 | 612 |

---

## Notes
This schema applies to all manually annotated images, regardless of capture device or source.  
Future revisions may include:
- `annotation_quality` (manual rating 1–5)  
- `valid_corners` (count of successfully annotated corners)  
- `annotator_name` (optional field for multi-user projects)  

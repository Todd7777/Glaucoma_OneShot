# Data Directory

## LAG Dataset Structure

Place the LAG dataset files in this directory:

```
data/
├── images/           # Fundus images (.jpg, .png)
├── labels.csv        # Image labels and metadata
└── README.md         # This file
```

## Labels Format

The `labels.csv` file should contain:
- `filename`: Image filename
- `label`: 0 (normal) or 1 (glaucoma)  
- `path`: Full path to image file

## Dataset Access

The LAG dataset requires permission from the original authors:
- Contact: ll1320@ic.ac.uk
- Reference: Li et al., CVPR 2019
- Download: [Dropbox link](https://www.dropbox.com/s/7mcngr3xhlaj5uc/LAG_database_part_1.rar?dl=0) (password required)

## Sample Data

For testing without the full dataset, run:
```bash
python main.py --create-sample-data
```

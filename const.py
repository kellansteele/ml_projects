# Define possible models in a dict.
# Format of the dict:
# option 1: model -> code
# option 2 – if model has multiple variants: model -> model variant -> code
MODELS = {
#    "YOLOv3": "YOLOv3",  # single model variant
    "Thermal YOLOv5": {  # multiple model variants
        "Batch size: 16, Epochs: 200": "models/exp1_16_200.pt",
        "Batch size: 16, Epochs: 400": "models/exp2_16_400.pt",
        "Batch size: 20, Epochs: 200": "models/exp3_20_200.pt",
        "Batch size: 20, Epochs: 300": "models/exp4_20_300.pt"
    },
}

# Define possible images in a dict.
IMAGES = {
    "Thermal YOLOv5": {
        "Image 1": "data/test/images/IMG_0033 2_jpg.rf.1ce012dff1ceb8d37c2ee1edd005843b.jpg",
        "Image 2": "data/test/images/IMG_0006 5_jpg.rf.cd46e6a862d6ffb7fce6795067ce7cc7.jpg",
        "Image 3": "data/test/images/IMG_0009_jpg.rf.ecdb212f7d7796e682a87e2e1d6e907e.jpg",
        "Image 4": "data/test/images/IMG_0113_jpg.rf.518ce21582555915f942463375a135b0.jpg",
        "Image 5": "data/test/images/IMG_0022_jpg.rf.c89662890a0f5d8a915677ed21165d2b.jpg",
        "Image 6": "data/test/images/IMG_0023 3_jpg.rf.ac45d9a3e591d1377f50b25c2415a5b7.jpg"
    },
}

# Define classes the model was trained over
CLASSES = {
    0:"dog",
    1:"person"
}
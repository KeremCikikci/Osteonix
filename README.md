# 🦴 Osteonix

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![YOLO](https://img.shields.io/badge/AI-YOLOv8%20/%20YOLOv11-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-BTXRD-purple.svg)

**Osteonix** is a professional desktop application built with PyQt5 for medical image analysis, specifically tailored to detect and manage bone tumors and lesions using deep learning (YOLO models trained on the **BTXRD** dataset).

---

## 🔬 Features

* **YOLO AI Detection Integration:** Automatically detects bone tumor classes (such as osteochondroma, giant cell tumor, osteosarcoma, etc.) on single images or in batch mode across an entire folder.
* **Interactive Bounding Box Editor:** Add, resize, move, delete, or customize bounding boxes (lesions) with real-time coordinate updates and color pickers.
* **Thumbnail & File Management:** Fast background thumbnail generation, batch renaming, sorting options, and drag-and-drop support.
* **Export Capabilities:** Export individual images or entire datasets with overlay bounding boxes burned directly into the output files.

---

## 🧠 Class Mapping (BTXRD Dataset)

The model detects and categorizes various bone conditions based on the BTXRD dataset classes:
* `0`: Osteochondroma
* `1`: Simple Bone Cyst
* `2`: Giant Cell Tumor
* `3`: Osteofibroma
* `4`: Other BT
* `5`: Osteosarcoma
* `6`: Other MT

---

## 🛠️ Technologies & Libraries

* **GUI Framework:** PyQt5
* **Computer Vision & AI:** Ultralytics YOLO, PyTorch, OpenCV/Qt Graphics View
* **Language:** Python

---
<img width="2861" height="1708" alt="Screenshot 2026-09-18 164547" src="https://github.com/user-attachments/assets/0ae22b5d-0c42-4f62-a2db-269a1c306d76" />

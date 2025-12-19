# 🎨 Real-Time Anime Style Conversion using Deep Learning

This project implements a **real-time anime-style image and video conversion system** using deep learning techniques. It captures live video from a webcam and converts each frame into an anime-style representation using **AnimeGANv2**, while displaying both the original and transformed outputs side by side.

---

## 📌 Project Overview

Anime style transfer has gained popularity in content creation, animation, and virtual avatars. This project demonstrates how **Generative Adversarial Networks (GANs)** can be used to transform real-world images and live camera feeds into anime-style visuals in real time.

The system:
- Captures frames from a webcam using OpenCV  
- Processes each frame using a pre-trained AnimeGANv2 model  
- Converts the visual style while preserving facial structure and expressions  
- Displays both original and anime-style video outputs simultaneously  

---

## 🧠 Model Used

- **AnimeGANv2**
- Loaded via **PyTorch Hub**
- Pre-trained model: `face_paint_512_v2`
- Automatically uses **GPU (CUDA)** if available, otherwise runs on CPU

---

## 🛠️ Technologies Used

- Python  
- PyTorch  
- AnimeGANv2  
- OpenCV  
- NumPy  
- PIL (Python Imaging Library)

---

## 📂 Project Structure

📁 Anime-Style-Conversion
│
├── deep_project.py # Main Python script
├── original_picture.jpeg # Sample original image
├── after_conversion.jpeg # Anime-style output image
├── README.md # Project documentation

pgsql
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/anime-style-conversion.git
cd anime-style-conversion
2️⃣ Install Required Libraries
bash
Copy code
pip install torch torchvision opencv-python pillow numpy
⚠️ For better performance, a CUDA-enabled GPU is recommended.

▶️ How to Run the Project
Run the main Python file:

bash
Copy code
python deep_project.py
Controls:
Webcam opens automatically

Two windows will appear:

Webcam - Original

Webcam - Anime

Press q to exit

🖼️ Sample Output
Original Image	Anime Converted Image
Real-world input image	Anime-style transformed output

(See original_picture.jpeg and after_conversion.jpeg for reference)

🚀 Applications
Anime and cartoon content creation

Virtual avatars and filters

Gaming and animation

Augmented and virtual reality

Creative AI applications

🔮 Future Enhancements
Custom anime dataset training

Performance analytics (FPS, latency)

Mobile or web-based deployment

Multi-style anime conversion

Integration with AR applications

📄 License
This project is for educational and research purposes only.

👨‍💻 Author
Alan TS
MSc Computer Science (Data Analytics)
Rajagiri College of Social Sciences, Kalamassery

⭐ If you find this project useful, feel free to star the repository!

yaml
Copy code

---

If you want, I can also:
- ✅ Customize this README for **college submission**
- ✅ Add **screenshots section**
- ✅ Write a **project report abstract**
- ✅ Make it **resume / LinkedIn ready**

Just tell me 👍







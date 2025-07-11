# 📝 Handwritten Digit Generator with DCGAN

A deep learning project to generate realistic handwritten digit images using a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset. This implementation demonstrates the power of GANs for synthetic image creation and is ideal for learning, experimentation, and dataset augmentation.

---

## 🚀 Features

- **Deep Convolutional GAN (DCGAN):**
  - Utilizes convolutional neural networks for both generator and discriminator.
- **MNIST Dataset:**
  - Trains on the popular MNIST dataset of handwritten digits (28x28 grayscale images).
- **Synthetic Digit Generation:**
  - Produces realistic, high-quality digit images from random noise vectors.
- **Training Visualization:**
  - Periodically saves and displays generated samples during training.
- **Customizable Hyperparameters:**
  - Easily adjust epochs, batch size, learning rates, and latent vector size.

---

## 📁 Project Structure

```bash
Handwritten-Digit-Generator---DCGAN/
│
├── handwritten_digit_generator.ipynb # Main Jupyter notebook
├── requirements.txt # Python dependencies (if provided)
└── README.md # Project documentation
```


---

## 🧑‍💻 Skills & Technologies

- Python, NumPy
- TensorFlow / Keras (deep learning)
- Matplotlib (visualization)
- DCGAN architecture (generator & discriminator)
- Data preprocessing and augmentation

---

## 🛠️ Getting Started

### 1. Environment Setup

- Clone or download the repository.
- Install dependencies (if requirements.txt is provided):

```bash
pip install -r requirements.txt
```


### 2. Run the Notebook

- Open `handwritten_digit_generator.ipynb` in Jupyter Notebook or Google Colab.
- Execute cells sequentially to:
- Load and preprocess MNIST data.
- Build and compile the generator and discriminator models.
- Train the DCGAN and visualize generated digits.

---

## 💡 Usage Highlights

- **Generate New Digits:**  
The trained generator can create new, never-before-seen handwritten digit images.
- **Experiment with Latent Space:**  
Modify the random noise input to explore how the generator responds.
- **Visualize Training Progress:**  
View sample outputs at intervals to monitor GAN learning.

---

## 📊 Example Results

| Epoch | Sample Output Description      |
|-------|-------------------------------|
| 1     | Noisy, unrecognizable digits  |
| 25    | Blurry digits, some structure |
| 50+   | Clear, realistic digits       |

---

## ❓ Troubleshooting

- **Training Instability:**  
GANs can be sensitive to hyperparameters. Try adjusting the learning rate or batch size if training diverges.
- **Slow Training:**  
Use GPU acceleration in Colab or a local machine with CUDA support for faster results.
- **Missing Packages:**  
Ensure all dependencies are installed as per the instructions.

---

## 📜 License

For educational and research purposes. Please cite the original authors and dataset if used in academic work.

---

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- DCGAN architecture: Radford, Metz, and Chintala (2015)[7]
- Open-source contributors for deep learning frameworks

---

Generate, visualize, and explore the world of handwritten digits with deep learning!

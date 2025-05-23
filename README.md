# Neural Network from Scratch

A comprehensive implementation of neural networks built from the ground up using pure Python and NumPy. This project demonstrates deep understanding of neural network fundamentals by implementing core components without relying on high-level frameworks like TensorFlow or PyTorch.

## 🧠 About

This repository contains a complete neural network implementation built from scratch, covering everything from basic perceptrons to multi-layer deep networks. The goal is to understand the mathematical foundations and inner workings of neural networks by implementing each component manually.

## 📁 Project Structure

```
neural-network-from-scratch/
├── activation_functions/     # Individual activation function classes
│   ├── relu.ipynb              # ReLU activation function
│   ├── sigmoid.ipynb           # Sigmoid activation function
│   ├── tanh.ipynb              # Tanh activation function
│   └── softmax.ipynb           # Softmax activation function
├── evaluations/             # Model evaluation metrics and utilities
├── implementations/         # Core neural network implementations
├── layers/                  # Different types of neural network layers
│   └── dense_layer.ipynb       # Fully connected layer implementation
├── loss_functions/          # Individual loss function classes
│   ├── mse.ipynb              # Mean Squared Error
│   └── cross_entropy.ipynb    # Cross-entropy loss
├── models/                  # Neural network model classes and architectures
│   └── neural_network.ipynb   # Main neural network class
├── optimizers/              # Individual optimizer classes
│   ├── sgd.ipynb              # Stochastic Gradient Descent
│   ├── adam.ipynb             # Adam optimizer
│   └── rmsprop.ipynb          # RMSprop optimizer
├── fashion_mnist.model      # Trained model on Fashion-MNIST dataset
├── fashion_mnist.parms      # Model parameters and weights
└── .gitignore
```

## 🚀 Features

### Core Components
- **Custom Neural Network Layers**: Dense/Fully connected layers with forward and backward propagation
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, and more
- **Loss Functions**: Mean Squared Error, Cross-entropy, and custom loss implementations
- **Optimizers**: Gradient Descent, Adam, RMSprop with learning rate scheduling
- **Model Evaluation**: Accuracy, precision, recall, F1-score metrics

### Advanced Features
- **Modular Architecture**: Easy to extend and customize components
- **Gradient Computation**: Manual backpropagation implementation
- **Model Serialization**: Save and load trained models
- **Fashion-MNIST Integration**: Complete example with real dataset

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/SamedyHUNX/neural-network-from-scratch
cd neural-network-from-scratch
```

2. Install required dependencies:
```bash
pip install numpy matplotlib pandas
```

## 📊 Performance

The implemented neural network achieves competitive performance on standard datasets:
- **Fashion-MNIST**: ~90% accuracy with optimized hyperparameters
- **Training Time**: Efficient implementation with vectorized operations

## 🔧 Components Deep Dive

### Activation Functions
- **ReLU**: Fast and effective for hidden layers
- **Sigmoid**: Classic activation for binary classification
- **Tanh**: Normalized sigmoid variant
- **Softmax**: Perfect for multi-class classification output

### Optimizers
- **Stochastic Gradient Descent (SGD)**: Basic optimization algorithm
- **Adam**: Adaptive learning rate with momentum
- **RMSprop**: Root mean square propagation

### Loss Functions
- **Mean Squared Error**: For regression problems
- **Cross-Entropy**: For classification tasks
- **Binary Cross-Entropy**: For binary classification

## 📚 Learning Resources

For detailed explanations, mathematical derivations, and step-by-step tutorials, check out my comprehensive Notion notebook:

🔗 **[Neural Network from Scratch - Complete Guide](https://chivalrous-kitty-57b.notion.site/Neural-Network-from-scratch-1f6dd392cfcb807fa4e3c50a68e88de3)**

This notebook covers:
- Mathematical foundations of neural networks
- Step-by-step implementation explanations
- Backpropagation algorithm derivation
- Optimization techniques and best practices

## 🤝 Connect

Feel free to reach out for discussions, collaborations, or questions:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://kh.linkedin.com/in/vadhna-samedy-hun-91a2b01b0)

## 🚧 Future Enhancements

- [ ] Convolutional Neural Network layers
- [ ] Recurrent Neural Network implementations
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] More advanced optimizers (AdamW, NAdam)
- [ ] GPU acceleration support
- [ ] More dataset examples

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Inspired by the need to understand neural networks from first principles
- Built as a learning exercise to master deep learning fundamentals
- Thanks to the open source community for mathematical resources and inspiration

---

⭐ **Star this repository if you find it helpful for learning neural networks!**

*Built with ❤️ and lots of mathematics*

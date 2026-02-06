import sys
import matplotlib.pyplot as plt
from final_project.data.mnist import load_mnist
from final_project.models.nn_scratch import NNClassifier

def run_demo():
    print("From-Scratch Neural Network Demo")
    print("Dataset: MNIST (784 dimensions, 10 classes)")
    print("Architecture: 784 -> 64 -> 10")
    print("Optimizer: Custom Adam Implementation")
    print("-" * 40)

    data = load_mnist() 
    
    model = NNClassifier(input_dim=784, hidden=64, num_classes=10, lr=0.005)
    
    print("\nStarting Training")
    model.fit(data.X_train, data.y_train, data.X_val, data.y_val, epochs=15, patience=2)
    

    print("\nGenerating training report")
    history = model.history
    plt.figure(figsize=(10, 5))
    plt.plot([x['train_loss'] for x in history], label='Train Loss', linewidth=2)
    plt.plot([x['val_loss'] for x in history], label='Val Loss', linewidth=2, linestyle='--')
    plt.title('Training Dynamics: NumPy vs MNIST')
    plt.xlabel('Epoch'); plt.ylabel('Cross Entropy Loss'); plt.legend(); grid=True
    
    output_path = 'mnist_training_curve.png'
    plt.savefig(output_path)
    print(f" Demo Complete. Loss curve saved to '{output_path}'.")

def main():
    run_demo()

if __name__ == "__main__":
    main()

import sys
import matplotlib.pyplot as plt
from final_project.data.titanic import load_titanic
from final_project.data.mnist import load_mnist
from final_project.models.nn_scratch import NNClassifier

def train_titanic():
    print("--- Running  Demo ---")
    data = load_titanic()
    model = NNClassifier(input_dim=data.X_train.shape[1], hidden=16, num_classes=2)
    model.fit(data.X_train, data.y_train, data.X_val, data.y_val, epochs=200)

def train_mnist():
    print("--- Running MNIST Demo (High-Dimensional) ---")
    # Load
    data = load_mnist() # 784 features
    
    # Initialize (784 -> 64 -> 10)
    model = NNClassifier(input_dim=784, hidden=64, num_classes=10, lr=0.005)
    
    # Train
    model.fit(data.X_train, data.y_train, data.X_val, data.y_val, epochs=15, patience=2)
    
    # Plot
    history = model.history
    plt.figure(figsize=(10, 5))
    plt.plot([x['train_loss'] for x in history], label='Train Loss')
    plt.plot([x['val_loss'] for x in history], label='Val Loss')
    plt.title('NumPy Neural Net on MNIST')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('mnist_training_curve.png')
    print("\n✅ Training complete. Plot saved to 'mnist_training_curve.png'.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m final_project.cli [titanic|mnist]")
        return
    
    command = sys.argv[1]
    if command == "titanic":
        train_titanic()
    elif command == "mnist":
        train_mnist()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()

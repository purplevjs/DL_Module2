from setup import *

def main():
    """
    Main function to execute the training processes based on user input.
    """
    print("=" * 80)
    print("SENTENCE ANALYSIS MODELS SETUP")
    print("=" * 80)
    print("\nAvailable training methods:")
    print("1. RoBERTa Model (standard training)")
    print("2. RoBERTa Model (cross-validation)")
    print("3. SVM Model")
    print("4. Naive Word-Matching Approach")
    print("5. Train All Models")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if 1 <= choice <= 5:
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    if choice == 1:
        print("\nTraining RoBERTa Model (standard training)...")
        train_roberta_model(cross_validation=False)
    elif choice == 2:
        print("\nTraining RoBERTa Model (cross-validation)...")
        n_splits = int(input("Enter number of folds for cross-validation (default=5): ") or 5)
        train_roberta_model(cross_validation=True, n_splits=n_splits)
    elif choice == 3:
        print("\nTraining SVM Model...")
        train_svm_model()
    elif choice == 4:
        print("\nTraining Naive Word-Matching Approach...")
        train_naive_approach()
    elif choice == 5:
        print("\nTraining All Models...")
        print("\n1. Training RoBERTa Model (standard training)...")
        train_roberta_model(cross_validation=False)
        print("\n2. Training SVM Model...")
        train_svm_model()
        print("\n3. Training Naive Word-Matching Approach...")
        train_naive_approach()
    
    print("\nAll training completed!")

if __name__ == "__main__":
    main()
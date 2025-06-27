from src.train import train_model
from transformers import BertTokenizerFast

if __name__ == "__main__":
    
    train_model()
    
    print("\nTraining complete. Model saved to 'model/' directory.")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    tokenizer.save_pretrained("model/")
    print("Tokenizer saved to 'model/'")





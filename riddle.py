#necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as
transforms
#teaching the Ai riddles and questions so that it can solve complex equations as it learns more and more
riddles = [
    "What has keys but can't open locks?",
    "What has a heart that doesn't beat?",
    "I have cities but no houses. I have mountains but not trees. I have water, but no fish. What am I?",
    "Using only addition, add 8's to get to get the number 1000.",
    "How can you add eight 4's so the total adds up to 500?"
]

answers = [
    "Piano",
    "Artichoke",
    "Map",
    "888+88+8+8+8",
    "444+44+4+4+4"
]
#Right now my Ai is a retard and doesn't understand english so I have to fix that now too.

# Tokenize the riddles and answers
tokenizer = get_tokenizer('basic_english')
riddles_tokens = [tokenizer(riddle) for riddle in riddles]
answers_tokens = [tokenizer(answer) for answer in answers]

# Create a vocabulary from the tokenized riddles and answers
def yield_tokens(data_iter):
    for data in data_iter:
        yield data

riddles_vocab = build_vocab_from_iterator(yield_tokens(riddles_tokens))
answers_vocab = build_vocab_from_iterator(yield_tokens(answers_tokens))

# Convert tokenized riddles and answers into tensors (numbers)
def text_pipeline(text):
    return torch.tensor([riddles_vocab[token] for token in tokenizer(text)], dtype=torch.long)

riddles_tensors = [text_pipeline(riddle) for riddle in riddles]
answers_tensors = [text_pipeline(answer) for answer in answers]

print(riddles_tensors)
print(answers_tensors)

#idk anymore
import torch
import torch.nn as nn
from transformers import BertModel

class RiddleSolver(nn.Module):
    def __init__(self):
        super(RiddleSolver, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Add a custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # BERT hidden size is 768
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(answers_vocab))  # Output size is the number of answers in the vocabulary
        )

    def forward(self, input_ids, attention_mask):
        # Pass the input through the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the output from the last hidden layer (CLS token)
        pooled_output = outputs.pooler_output

        # Pass the pooled output through the classifier
        logits = self.classifier(pooled_output)

        return logits

# Create an instance of the RiddleSolver model
model = RiddleSolver()
import torch.optim as optim

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is suitable for multi-class classification

# Choose an appropriate optimizer and learning rate
optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # AdamW optimizer is recommended for transformer models

# If you have access to a GPU, move the model and loss function to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)
# Step 6: Train the AI Model
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Zero the gradients to avoid accumulation
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track total loss and number of correct predictions
            total_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += input_ids.size(0)

        # Print the training statistics after each epoch
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Train the model for a specified number of epochs
num_epochs = 5
train_model(model, dataloader, criterion, optimizer, num_epochs)
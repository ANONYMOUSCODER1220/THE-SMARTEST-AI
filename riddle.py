class RiddleDataset(Dataset):
    def __init__(self, riddles, answers, tokenizer, riddles_vocab, answers_vocab):
        self.riddles = riddles
        self.answers = answers
        self.tokenizer = tokenizer
        self.riddles_vocab = riddles_vocab
        self.answers_vocab = answers_vocab

    def __len__(self):
        return len(self.riddles)

    def __getitem__(self, idx):
        riddle = self.riddles[idx]
        answer = self.answers[idx]

        riddle_tensor = text_pipeline(riddle)
        answer_tensor = text_pipeline(answer)

        # Create a dictionary with riddle, answer, and label (answer index)
        return {
            'riddle': riddle_tensor,
            'answer': answer_tensor,
            'label': answers_vocab[answer]  # Use answer vocabulary index as label
        }

class RiddleSolver(nn.Module):
    def __init__(self):
        super(RiddleSolver, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Adjust hidden size based on the BERT model used
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(answers_vocab))  # Output size is vocabulary size for answer classification
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

import pandas as pd
import numpy as np
import random
from src.config import RAW_DATA_PATH

def generate_synthetic_data(n_samples=2000):
    """
    Generates a synthetic email dataset for Spam/Phishing Detection.
    """
    print(f"Generating synthetic email dataset with {n_samples} samples...")
    
    spam_subjects = [
        "Win a free iPhone!", "Urgent: Your account is locked", 
        "Claim your lottery prize", "Cheap medication online", 
        "Earn $5000 a week from home", "Final Notice: Invoice Overdue",
        "You have a new secure message", "Congratulations! You're a Winner!"
    ]
    
    ham_subjects = [
        "Meeting at 10 AM", "Project update", "Lunch tomorrow?", 
        "Invoice for services", "Let's catch up", "Weekly Status Report",
        "Code review request", "Notes from yesterday's call"
    ]
    
    spam_bodies = [
        "Click here to claim your prize now! Limited time offer. Unsubscribe here.",
        "Your bank account has been suspended due to suspicious activity. Verify your identity immediately by clicking the link.",
        "You have been selected to win a $1000 gift card! Click the link to claim.",
        "Make money fast working from home! No experience needed. Reply to this email.",
        "Dear Customer, your password will expire in 24 hours. Click below to update it.",
        "Exclusive offer just for you! Buy one get one free. Don't miss out."
    ]
    
    ham_bodies = [
        "Hi team, let's schedule a meeting to discuss the new project. Thanks.",
        "Attached is the invoice for last month's work. Let me know if you have any questions.",
        "Are we still on for lunch tomorrow? Let me know where you want to go.",
        "I've reviewed the PR and it looks good. Ready to merge whenever you are.",
        "Just checking in on the status of the report. Do you need any help?",
        "It was great catching up yesterday. Let's do it again soon."
    ]
    
    data = []
    # Generate imbalanced dataset (e.g., 20% spam, 80% ham)
    for _ in range(n_samples):
        is_spam = np.random.rand() < 0.2
        if is_spam:
            subj = random.choice(spam_subjects)
            body = random.choice(spam_bodies)
            label = 1
        else:
            subj = random.choice(ham_subjects)
            body = random.choice(ham_bodies)
            label = 0
            
        # Add some random noise to make texts slightly unique
        noise = str(random.randint(100, 999))
        if random.random() > 0.5:
            subj += f" [{noise}]"
        
        data.append({"Subject": subj, "Body": body, "Class": label})
        
    df = pd.DataFrame(data)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()

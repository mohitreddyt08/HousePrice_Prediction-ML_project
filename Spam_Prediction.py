from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

messages = [
    # Spam messages
    "Win a free prize now",
    "Claim your free gift",
    "Congratulations you won money",
    "Click here to claim your reward",
    "Free offer limited time",
    "You won a cash prize",
    "Get free coupons today",
    "Urgent claim your reward",
    "Congratulations you have won a lottery",
    "Claim your cash bonus now",
    "Exclusive offer buy now",
    "You are a lucky winner",
    "Get a free gift card",
    "Earn money quickly with this offer",
    "Your prize is waiting claim now",
    "Limited time discount offer",
    "Free entry to win a prize",
    "You have won a special reward",
    "Claim your bonus today",
    "Congratulations claim your gift",

    # Not-spam messages
    "Hey are you coming to class",
    "Let's meet tomorrow",
    "Please send me the notes",
    "Are we going to college today",
    "I will call you later",
    "Did you finish your homework",
    "Thanks for helping me",
    "See you in the evening",
    "Can you help me with Python",
    "What time is the lecture",
    "Please share the assignment",
    "I will meet you at college",
    "Have you completed the project",
    "Let's study together tomorrow",
    "Can you send the class notes",
    "I am going home now",
    "Are you free this evening",
    "Please call me when you arrive",
    "We have an exam next week",
    "Thank you for your help"
]

labels = ["spam"] * 20 + ["not spam"] * 20

X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

model = make_pipeline(
    CountVectorizer(lowercase=True),
    MultinomialNB()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Test accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

while True:
    message = input("\nEnter a message (or type 'exit'): ")

    if message.lower() == "exit":
        break

    result = model.predict([message])
    probabilities = model.predict_proba([message])[0]

    print("Prediction:", result[0])
    print("Spam probability:",
          round(probabilities[list(model.classes_).index("spam")] * 100, 2),
          "%")
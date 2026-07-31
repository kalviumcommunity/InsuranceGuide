import os
import tiktoken

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

# GPT tokenizer
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# Pricing (Example Prices)
# Change these if using another model.

INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "token_count_results.txt")


# ----------------------------------------------------
# Function to count tokens
# ----------------------------------------------------

def count_tokens(text):
    tokens = encoding.encode(text)
    return len(tokens)


# ----------------------------------------------------
# Function to calculate cost
# ----------------------------------------------------

def calculate_input_cost(tokens):
    return (tokens / 1_000_000) * INPUT_PRICE_PER_MILLION


def calculate_output_cost(tokens):
    return (tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION


# ----------------------------------------------------
# Function to read one document from data folder
# ----------------------------------------------------

def read_sample_document():

    if not os.path.exists(DATA_FOLDER):
        return ""

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".txt"):

            path = os.path.join(DATA_FOLDER, file)

            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    return ""


# ----------------------------------------------------
# Sample Texts
# ----------------------------------------------------

sample1 = "What insurance plans cover accidental hospitalization?"

sample2 = """
Health insurance provides financial protection against medical expenses.
Policies may include hospitalization costs, pre and post hospitalization
expenses, day-care procedures, ambulance charges, and cashless treatment
at network hospitals.
"""

sample3 = read_sample_document()

if sample3 == "":
    sample3 = """
Insurance policies help individuals reduce financial risks by transferring
potential losses to an insurance provider. Different plans provide coverage
for health, vehicles, homes, travel, and life. Customers should compare
premium amounts, waiting periods, exclusions, and claim settlement ratios
before selecting a policy.
""" * 10


samples = [
    ("Short Question", sample1),
    ("Paragraph", sample2),
    ("Full Document", sample3)
]


# ----------------------------------------------------
# Create output directory
# ----------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ----------------------------------------------------
# Process Samples
# ----------------------------------------------------

results = []

print("=" * 70)
print("TOKEN COUNT REPORT")
print("=" * 70)

for title, text in samples:

    characters = len(text)

    words = len(text.split())

    tokens = count_tokens(text)

    input_cost = calculate_input_cost(tokens)

    output_cost = calculate_output_cost(tokens)

    report = f"""
------------------------------------------------------------
{title}

Characters      : {characters}
Words           : {words}
Tokens          : {tokens}

Estimated Input Cost  : ${input_cost:.8f}
Estimated Output Cost : ${output_cost:.8f}
"""

    print(report)

    results.append(report)

print("=" * 70)

# ----------------------------------------------------
# Character vs Token Relationship
# ----------------------------------------------------

relationship = """

============================================================
Character vs Token Relationship
============================================================

Characters and token counts generally increase together.

However, they are NOT exactly proportional.

Reasons:

1. Long words may be split into multiple tokens.

2. Numbers and punctuation create different token patterns.

3. URLs are usually broken into many tokens.

4. Programming code often generates more tokens than plain English.

5. Different languages (Hindi, Chinese, Japanese, etc.)
   have different tokenization behaviour.

This is why measuring token usage is important before
building a production RAG application.
"""

print(relationship)

results.append(relationship)

# ----------------------------------------------------
# Save Results
# ----------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("TOKEN COUNT REPORT\n")
    file.write("=" * 60)

    for item in results:
        file.write(item)

print(f"\nResults saved to: {OUTPUT_FILE}")
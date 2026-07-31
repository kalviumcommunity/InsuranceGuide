import os
import tiktoken

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

MAX_TOKENS = 300

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "history_demo.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -------------------------------------------------------
# Count Tokens
# -------------------------------------------------------

def count_tokens(text):
    return len(encoding.encode(text))


def total_history_tokens(history):
    total = 0

    for message in history:
        total += count_tokens(message["content"])

    return total


# -------------------------------------------------------
# Trim History
# -------------------------------------------------------

def trim_history(history):

    while total_history_tokens(history) > MAX_TOKENS and len(history) > 3:

        print("\nToken limit exceeded.")
        log.append("\nToken limit exceeded.")

        removed_user = history.pop(1)
        removed_assistant = history.pop(1)

        print(f"Removed User      : {removed_user['content'][:50]}...")
        print(f"Removed Assistant : {removed_assistant['content'][:50]}...")

        log.append(f"Removed User      : {removed_user['content']}")
        log.append(f"Removed Assistant : {removed_assistant['content']}")

        print(
            f"Tokens After Trim : {total_history_tokens(history)}"
        )

        log.append(
            f"Tokens After Trim : {total_history_tokens(history)}"
        )


# -------------------------------------------------------
# Conversation History
# -------------------------------------------------------

history = [

    {
        "role": "system",
        "content": "You are an intelligent insurance assistant that answers customer questions accurately."
    }

]


# -------------------------------------------------------
# Demo Conversation
# -------------------------------------------------------

conversation = [

    ("What is health insurance?",
     "Health insurance helps pay for medical expenses."),

    ("Does it cover surgeries?",
     "Most policies cover surgeries based on policy conditions."),

    ("What is a waiting period?",
     "A waiting period is the time before certain benefits become active."),

    ("Can I insure my parents?",
     "Yes. Many insurers offer family floater and senior citizen plans."),

    ("What is cashless treatment?",
     "Cashless treatment allows hospitals to settle bills directly with insurers."),

    ("What documents are needed for claims?",
     "Identity proof, hospital bills, discharge summary and policy details."),

    ("Does insurance cover dental treatment?",
     "Usually only if specifically included in the policy."),

    ("Can pre-existing diseases be covered?",
     "Yes, after the applicable waiting period."),

    ("What is a deductible?",
     "A deductible is the amount you pay before insurance starts paying."),

    ("Can I renew my policy every year?",
     "Yes, most health insurance policies are renewable annually."),

    ("What is a network hospital?",
     "A hospital partnered with the insurance company."),

    ("Can I buy insurance online?",
     "Yes. Most insurers provide online purchase and renewal facilities."),

    ("What is claim settlement ratio?",
     "It represents the percentage of claims settled by an insurer."),

    ("How do I compare insurance plans?",
     "Compare premiums, coverage, exclusions, waiting periods and settlement ratio.")

]


log = []

print("=" * 70)
print("CHAT HISTORY DEMO")
print("=" * 70)

log.append("=" * 70)
log.append("CHAT HISTORY DEMO")
log.append("=" * 70)


# -------------------------------------------------------
# Simulate Conversation
# -------------------------------------------------------

for user_message, assistant_message in conversation:

    history.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

    current_tokens = total_history_tokens(history)

    print("\n--------------------------------------------")
    print(f"User      : {user_message}")
    print(f"Assistant : {assistant_message}")
    print(f"Current Tokens : {current_tokens}")

    log.append("\n--------------------------------------------")
    log.append(f"User      : {user_message}")
    log.append(f"Assistant : {assistant_message}")
    log.append(f"Current Tokens : {current_tokens}")

    if current_tokens > MAX_TOKENS:
        trim_history(history)


print("\nConversation completed successfully.")
print(f"Final Token Count : {total_history_tokens(history)}")

log.append("\nConversation completed successfully.")
log.append(f"Final Token Count : {total_history_tokens(history)}")


# -------------------------------------------------------
# Save Output
# -------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for line in log:
        file.write(line + "\n")


print(f"\nHistory demo saved to {OUTPUT_FILE}")
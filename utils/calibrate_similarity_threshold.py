"""
Calibration script: measures REAL cosine similarity scores between
paper summary pairs so you can pick an evidence-based
SIMILARITY_THRESHOLD for agents/contradiction_detector.py instead of
trusting the placeholder estimate in that file.

Why this exists: the sandbox used to build this project has no
network access to huggingface.co, so the threshold in
contradiction_detector.py is a reasoned guess, not a measured value.
This script lets you check real numbers on your own machine, ideally
against your own actual paper summaries once you have a few.

Usage:
    uv run python utils/calibrate_similarity_threshold.py
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Replace these with REAL summaries from your own pipeline once you
# have a batch of papers processed -- these examples are illustrative
# placeholders to get you a first reading.
EXAMPLE_PAIRS = {
    "same_topic_agreeing": (
        "TinyBERT uses knowledge distillation to compress BERT, achieving 96% of BERT "
        "performance with 7.5x fewer parameters via attention and hidden state transfer.",
        "DistilBERT applies knowledge distillation to reduce BERT model size by 40% while "
        "retaining 97% of its language understanding capabilities, using a similar "
        "teacher-student framework."
    ),
    "same_topic_contradicting": (
        "Our experiments show that removing the Next Sentence Prediction objective during "
        "BERT pretraining IMPROVES downstream task performance by 1.2 points on average.",
        "We find that the Next Sentence Prediction objective is critical to BERT "
        "pretraining; removing it DEGRADES downstream performance by up to 3 points "
        "across GLUE tasks."
    ),
    "unrelated_topics": (
        "TinyBERT uses knowledge distillation to compress BERT for faster NLP inference.",
        "This paper proposes a new reinforcement learning algorithm for robotic arm "
        "manipulation in cluttered environments using vision-based grasping."
    ),
}


def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Pairwise similarity scores on illustrative examples:")
    print("(Replace EXAMPLE_PAIRS with your own real summaries for a meaningful calibration)")
    print()

    for label, (text_a, text_b) in EXAMPLE_PAIRS.items():
        embeddings = encoder.encode([text_a, text_b])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"  {label:30s} -> {score:.3f}")

    print()
    print("How to use this:")
    print("- 'same_topic_contradicting' should ideally score LOWER than 'same_topic_agreeing'")
    print("  if similarity alone is a useful contradiction signal.")
    print("- If contradicting and agreeing pairs score similarly (both high), it confirms")
    print("  embedding similarity mostly captures TOPIC, not AGREEMENT -- meaning the")
    print("  threshold should be set fairly high (e.g. 0.6-0.7) to send more same-topic")
    print("  pairs to the LLM for actual verification, rather than relying on a low")
    print("  threshold to pre-filter for you.")
    print("- 'unrelated_topics' should score clearly lower than both -- if not, something")
    print("  is off with the embedding setup.")


if __name__ == "__main__":
    main()
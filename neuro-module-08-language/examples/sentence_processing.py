"""
Sentence Processing Demo

Demonstrates hierarchical language processing through
the 4-level hierarchy: Phonological → Syntactic → Semantic → Pragmatic
"""

import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.predictive_language import PredictiveLanguageProcessor
from src.recursive_parser import RecursiveGrammar, ConstituentParser, compare_human_vs_llm


def demo_sentence_processing():
    """Process sentences through the hierarchy"""
    print("=" * 60)
    print("SENTENCE PROCESSING DEMO")
    print("4-Level Hierarchy: Phonological → Syntactic → Semantic → Pragmatic")
    print("=" * 60)

    processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["a", "big", "dog", "ran", "quickly"],
        ["she", "thinks", "that", "he", "knows"],
    ]

    for i, sentence in enumerate(sentences, 1):
        print(f"\n--- Sentence {i}: '{' '.join(sentence)}' ---")

        result = processor.process_utterance(sentence)

        print(f"  Tokens processed: {result['tokens']}")
        print(f"  Total prediction error: {result['total_error']:.4f}")

        timescales = result['layer_timescales']
        print(f"\n  Layer timescales (confirms hierarchy):")
        print(f"    Phonological: {timescales[0]*1000:.0f}ms")
        print(f"    Syntactic:    {timescales[1]*1000:.0f}ms")
        print(f"    Semantic:     {timescales[2]*1000:.0f}ms")
        print(f"    Pragmatic:    {timescales[3]*1000:.0f}ms")

        print(f"\n  Per-token processing:")
        for j, token_result in enumerate(result['token_results']):
            print(f"    '{sentence[j]}': error={token_result.total_error:.3f}, "
                  f"Broca inhibition={token_result.broca_inhibition:.3f}")

        processor.reset()


def demo_hierarchical_vs_linear():
    """Compare human-like hierarchical parsing with LLM-style linear prediction"""
    print("\n" + "=" * 60)
    print("HUMAN vs LLM LANGUAGE PROCESSING")
    print("Key difference: Hierarchy vs Linear prediction")
    print("=" * 60)

    sentence = "the cat sat on the mat"
    result = compare_human_vs_llm(sentence)

    print(f"\nInput: '{sentence}'")
    print(f"\n--- Human Processing (Hierarchical) ---")
    print(f"  Has hierarchical structure: {result['human_has_hierarchy']}")
    print(f"  Tree depth: {result['human_depth']}")
    print(f"\n  Constituent structure:")
    print(result['human_structure'])

    print(f"\n--- LLM Processing (Linear) ---")
    print(f"  Has hierarchical structure: {result['llm_has_hierarchy']}")
    print(f"  Processing type: Next-token prediction")
    print(f"\n  Token predictions (each predicts next):")
    tokens = sentence.split()
    for i, pred in enumerate(result['llm_predictions']):
        context = ' '.join(tokens[:i+1])
        print(f"    Given '{context}' → predicted '{pred}'")

    print(f"\n--- Key Difference ---")
    print(f"  {result['key_difference']}")


def demo_recursive_structure():
    """Demonstrate recursive phrase structure"""
    print("\n" + "=" * 60)
    print("RECURSIVE CONSTITUENT STRUCTURE")
    print("Context-free grammar generates hierarchical trees")
    print("=" * 60)

    grammar = RecursiveGrammar()

    print("\n--- Grammar Rules ---")
    for lhs, rules in list(grammar.rules.items())[:4]:
        rule_strs = [' '.join(r) for r in rules]
        print(f"  {lhs} → {' | '.join(rule_strs)}")
    print("  ...")

    print("\n--- Generated Sentences ---")
    for i in range(3):
        tree = grammar.generate('S', max_depth=4)
        terminals = tree.get_terminals()
        print(f"\n  {i+1}. '{' '.join(terminals)}'")
        print(f"     Depth: {tree.depth()}, Size: {tree.size()} nodes")
        print(f"     Structure:")
        for line in str(tree).split('\n')[:6]:
            print(f"       {line}")
        if tree.depth() > 3:
            print("       ...")


def demo_layer_activations():
    """Show activations at each layer"""
    print("\n" + "=" * 60)
    print("LAYER ACTIVATIONS")
    print("Different timescales capture different linguistic levels")
    print("=" * 60)

    processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

    sentence = ["the", "quick", "brown", "fox"]

    print(f"\nProcessing: '{' '.join(sentence)}'")
    result = processor.process_utterance(sentence)

    print("\n--- Final Layer States ---")
    activations = processor.get_layer_activations()

    for layer_name, activation in activations.items():
        mean_act = np.mean(np.abs(activation))
        max_act = np.max(np.abs(activation))
        print(f"\n  {layer_name.capitalize()}:")
        print(f"    Shape: {activation.shape}")
        print(f"    Mean |activation|: {mean_act:.4f}")
        print(f"    Max |activation|: {max_act:.4f}")


if __name__ == '__main__':
    demo_sentence_processing()
    demo_hierarchical_vs_linear()
    demo_recursive_structure()
    demo_layer_activations()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key findings demonstrated:
1. Language processing has hierarchical structure (phonological → pragmatic)
2. Each level operates at different timescales (10ms → 1000ms)
3. Humans use recursive constituent structure; LLMs use linear prediction
4. Predictions flow top-down while evidence flows bottom-up
""")

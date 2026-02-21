"""
Recursive Parser - Hierarchical Constituent Structure

Implements hierarchical phrase structure parsing that distinguishes
human language from LLM-style linear prediction.

Key differences:
- Human: Hierarchical constituent structure, recursive generative function
- LLM: Linear sequence prediction, probabilistic token sampling
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class PhraseType(Enum):
    """Types of syntactic phrases"""
    S = "S"       # Sentence
    NP = "NP"     # Noun Phrase
    VP = "VP"     # Verb Phrase
    PP = "PP"     # Prepositional Phrase
    AP = "AP"     # Adjective Phrase
    ADVP = "ADVP" # Adverb Phrase
    CP = "CP"     # Complementizer Phrase
    IP = "IP"     # Inflectional Phrase
    DP = "DP"     # Determiner Phrase


@dataclass
class Constituent:
    """Tree node in phrase structure

    Represents a syntactic constituent with hierarchical structure.
    """
    label: str
    children: List['Constituent'] = field(default_factory=list)
    head: Optional[Union[str, 'Constituent']] = None
    features: Dict[str, any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if this is a terminal (leaf) node"""
        return len(self.children) == 0

    def depth(self) -> int:
        """Compute depth of subtree"""
        if self.is_terminal():
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """Count total nodes in subtree"""
        if self.is_terminal():
            return 1
        return 1 + sum(child.size() for child in self.children)

    def get_terminals(self) -> List[str]:
        """Get all terminal nodes (words)"""
        if self.is_terminal():
            return [self.label]
        terminals = []
        for child in self.children:
            terminals.extend(child.get_terminals())
        return terminals

    def to_string(self, indent: int = 0) -> str:
        """Pretty print the tree"""
        result = " " * indent + f"[{self.label}"
        if self.is_terminal():
            result += "]"
        else:
            result += "\n"
            for child in self.children:
                result += child.to_string(indent + 2) + "\n"
            result += " " * indent + "]"
        return result

    def __str__(self) -> str:
        return self.to_string()


class RecursiveGrammar:
    """Recursive generative grammar

    Implements context-free grammar rules for generating
    and parsing hierarchical structures.
    """

    def __init__(self):
        # CFG-style rules: LHS -> [RHS options]
        self.rules: Dict[str, List[List[str]]] = {
            'S': [['NP', 'VP'], ['CP']],
            'NP': [['Det', 'N'], ['Det', 'AP', 'N'], ['N'], ['NP', 'PP']],
            'VP': [['V'], ['V', 'NP'], ['V', 'NP', 'PP'], ['V', 'CP']],
            'PP': [['P', 'NP']],
            'AP': [['Adj'], ['Adv', 'Adj']],
            'CP': [['C', 'S']],
        }

        # Terminal categories
        self.terminals = {'Det', 'N', 'V', 'P', 'Adj', 'Adv', 'C'}

        # Sample lexicon
        self.lexicon: Dict[str, List[str]] = {
            'Det': ['the', 'a', 'this', 'that'],
            'N': ['cat', 'dog', 'man', 'woman', 'book', 'idea'],
            'V': ['saw', 'likes', 'thinks', 'said', 'believes'],
            'P': ['in', 'on', 'with', 'to', 'from'],
            'Adj': ['big', 'small', 'happy', 'sad', 'interesting'],
            'Adv': ['very', 'quite', 'extremely'],
            'C': ['that', 'if', 'whether'],
        }

    def generate(self, start: str = 'S', max_depth: int = 5) -> Constituent:
        """Generate a random sentence structure

        Uses recursive expansion of grammar rules.
        """
        if max_depth <= 0 or start in self.terminals:
            # Terminal: pick a word
            if start in self.lexicon:
                word = np.random.choice(self.lexicon[start])
                return Constituent(label=word, head=word)
            return Constituent(label=start, head=start)

        if start not in self.rules:
            return Constituent(label=start, head=start)

        # Non-terminal: expand recursively
        rule_options = self.rules[start]
        chosen_rule = rule_options[np.random.randint(len(rule_options))]

        children = []
        for symbol in chosen_rule:
            child = self.generate(symbol, max_depth - 1)
            children.append(child)

        # Determine head (simplified: first non-function word)
        head = children[0] if children else None

        return Constituent(label=start, children=children, head=head)

    def is_grammatical(self, tree: Constituent) -> bool:
        """Check if tree conforms to grammar rules"""
        if tree.is_terminal():
            return True

        # Check if this expansion is valid
        if tree.label in self.rules:
            child_labels = [c.label for c in tree.children]

            # Check if children match any rule
            for rule in self.rules[tree.label]:
                if self._matches_rule(child_labels, rule):
                    # Recursively check children
                    return all(self.is_grammatical(c) for c in tree.children)

        return False

    def _matches_rule(self, children: List[str], rule: List[str]) -> bool:
        """Check if children match a rule"""
        if len(children) != len(rule):
            return False

        for child, expected in zip(children, rule):
            # Child can be the expected category or a word from that category
            if child == expected:
                continue
            if expected in self.lexicon and child in self.lexicon[expected]:
                continue
            return False

        return True

    def add_rule(self, lhs: str, rhs: List[str]) -> None:
        """Add a new grammar rule"""
        if lhs not in self.rules:
            self.rules[lhs] = []
        self.rules[lhs].append(rhs)

    def add_lexical_item(self, category: str, word: str) -> None:
        """Add a word to the lexicon"""
        if category not in self.lexicon:
            self.lexicon[category] = []
        self.lexicon[category].append(word)


class ConstituentParser:
    """Parser that builds hierarchical structure

    Uses shift-reduce parsing to construct constituent trees.
    """

    def __init__(self, grammar: RecursiveGrammar):
        self.grammar = grammar
        self.stack: List[Constituent] = []
        self.buffer: List[str] = []

    def parse(self, tokens: List[str]) -> Optional[Constituent]:
        """Parse tokens into constituent structure"""
        self.stack = []
        self.buffer = list(reversed(tokens))  # Right to left for shifting

        while self.buffer or len(self.stack) > 1:
            # Try to reduce first
            if self._try_reduce():
                continue

            # Otherwise shift
            if self.buffer:
                self._shift()
            else:
                # Can't reduce or shift
                break

        if len(self.stack) == 1:
            return self.stack[0]

        # Partial parse
        return Constituent(label='S', children=self.stack)

    def _shift(self) -> None:
        """Move token from buffer to stack"""
        if self.buffer:
            token = self.buffer.pop()
            # Determine category
            category = self._get_category(token)
            constituent = Constituent(label=category, head=token)
            self.stack.append(constituent)

    def _get_category(self, word: str) -> str:
        """Look up word category"""
        for cat, words in self.grammar.lexicon.items():
            if word in words:
                return cat
        return word  # Unknown word becomes its own category

    def _try_reduce(self) -> bool:
        """Try to reduce top of stack using grammar rules"""
        if len(self.stack) < 1:
            return False

        # Try different reduction lengths
        for length in range(min(4, len(self.stack)), 0, -1):
            items = self.stack[-length:]
            labels = [item.label for item in items]

            # Check all rules
            for lhs, rules in self.grammar.rules.items():
                for rule in rules:
                    if labels == rule:
                        # Reduce!
                        self.stack = self.stack[:-length]
                        new_constituent = Constituent(
                            label=lhs,
                            children=items,
                            head=items[0]
                        )
                        self.stack.append(new_constituent)
                        return True

        return False


class LinearPredictor:
    """LLM-style linear prediction (for comparison)

    Implements probabilistic token prediction without
    hierarchical structure - the way LLMs process language.
    """

    def __init__(self, vocab_size: int = 1000, context_size: int = 10, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.hidden_dim = hidden_dim

        # Simple linear model
        self.W_embed = np.random.randn(vocab_size, hidden_dim) * 0.1
        # W_context projects from context embedding to hidden
        self.W_context = np.random.randn(hidden_dim, hidden_dim * context_size) * 0.1
        self.W_output = np.random.randn(hidden_dim, vocab_size) * 0.1

        # Context window
        self.context: List[int] = []

        # Token to ID mapping
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.next_id = 0

    def _get_token_id(self, token: str) -> int:
        """Get or create ID for token"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id = min(self.next_id + 1, self.vocab_size - 1)
        return self.token_to_id[token]

    def predict_next(self, context: List[str]) -> Tuple[str, np.ndarray]:
        """Predict next token given context

        This is LINEAR prediction - no hierarchical structure.
        Just P(next | previous tokens).

        Returns:
            Tuple of (predicted token, probability distribution)
        """
        # Encode context
        context_ids = [self._get_token_id(t) for t in context[-self.context_size:]]

        # Pad if needed
        while len(context_ids) < self.context_size:
            context_ids.insert(0, 0)

        # Embed context
        embeddings = []
        for token_id in context_ids:
            emb = self.W_embed[token_id % self.vocab_size]
            embeddings.append(emb)

        context_vec = np.concatenate(embeddings)

        # Project to hidden
        hidden = np.tanh(self.W_context @ context_vec)

        # Output distribution
        logits = self.W_output.T @ hidden
        probs = self._softmax(logits)

        # Sample or take argmax
        predicted_id = np.argmax(probs)
        predicted_token = self.id_to_token.get(predicted_id, '<unk>')

        return predicted_token, probs

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-8)

    def generate_sequence(self, start: List[str], length: int = 10) -> List[str]:
        """Generate sequence by iterative prediction"""
        sequence = list(start)

        for _ in range(length):
            next_token, _ = self.predict_next(sequence)
            sequence.append(next_token)

        return sequence

    def has_hierarchical_structure(self) -> bool:
        """Linear predictor has NO hierarchical structure"""
        return False


def compare_human_vs_llm(sentence: str) -> Dict[str, any]:
    """Compare human-like parsing with LLM-style prediction

    Demonstrates the key difference:
    - Human: Hierarchical constituent structure
    - LLM: Linear sequence prediction
    """
    tokens = sentence.split()

    # Human-like: recursive constituent parsing
    grammar = RecursiveGrammar()
    parser = ConstituentParser(grammar)

    # Generate a tree (for demonstration)
    tree = grammar.generate('S', max_depth=4)

    # LLM-like: linear prediction
    linear = LinearPredictor()
    predictions = []
    for i in range(len(tokens)):
        pred, probs = linear.predict_next(tokens[:i+1])
        predictions.append(pred)

    return {
        'input': sentence,
        'human_structure': tree,
        'human_has_hierarchy': True,
        'human_depth': tree.depth(),
        'llm_predictions': predictions,
        'llm_has_hierarchy': False,
        'key_difference': "Human language builds recursive tree structures; LLMs predict next token linearly"
    }

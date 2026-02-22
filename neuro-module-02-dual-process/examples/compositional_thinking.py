"""
Demo: Compositional Thinking - "Jump Twice"

Demonstrates the HPC-PFC complex enabling novel combinations
of known concepts.

Based on research showing:
- Hippocampus rapidly encodes episodic experiences
- PFC extracts generalizable schemas
- Together they enable composing known concepts in novel ways
"""

import sys

sys.path.insert(0, "..")

from src.hpc_pfc_complex import HPCPFCComplex
from src.system2.relational_reasoning import RelationalReasoning, RelationType


def demo_basic_composition():
    """Basic concept composition: jump + twice"""
    print("=" * 60)
    print("DEMO 1: Basic Composition - 'Jump Twice'")
    print("=" * 60)

    rr = RelationalReasoning()

    # Create known concepts
    jump = rr.create_element(
        content="jump",
        type_tag="action",
        features={"repeatable": True, "physical": True, "discrete": True},
    )

    twice = rr.create_element(
        content=2, type_tag="modifier", features={"modifier_type": "repetition", "value": 2}
    )

    print(f"Concept 1: {jump.content} (type: {jump.type_tag})")
    print(f"Concept 2: {twice.content} (type: {twice.type_tag})")

    # Bind them together
    jump_twice = rr.bind_modifier(jump, twice)

    print(f"\nComposed structure:")
    print(f"  Elements: {[e.content for e in jump_twice.elements.values()]}")
    print(f"  Relation: {jump_twice.relations[0].relation_type.value}")

    # This structure can now be "executed" - repeat jump action
    print(f"\nInterpretation: Perform 'jump' action {twice.content} times")
    print("✓ Novel combination created from known parts")
    print()


def demo_action_with_roles():
    """Action frame composition with agent and patient"""
    print("=" * 60)
    print("DEMO 2: Action Frame - 'The dog bit the cat'")
    print("=" * 60)

    rr = RelationalReasoning()

    # Create elements
    dog = rr.create_element("dog", type_tag="entity", features={"animate": True})
    cat = rr.create_element("cat", type_tag="entity", features={"animate": True})
    bite = rr.create_element("bite", type_tag="action", features={"transitive": True})

    print("Elements:")
    print(f"  Action: {bite.content}")
    print(f"  Agent: {dog.content}")
    print(f"  Patient: {cat.content}")

    # Create action structure with roles
    structure = rr.create_action_structure(action=bite, agent=dog, patient=cat)

    print(f"\nAction frame structure:")
    print(f"  Elements: {list(structure.elements.keys())}")
    for rel in structure.relations:
        source = structure.elements.get(rel.source)
        target = structure.elements.get(rel.target)
        print(
            f"  {source.content if source else '?'} --[{rel.relation_type.value}]--> "
            f"{target.content if target else '?'}"
        )

    print("✓ Complex sentence structure represented compositionally")
    print()


def demo_nested_composition():
    """Nested composition: (jump twice) quickly"""
    print("=" * 60)
    print("DEMO 3: Nested Composition - '(Jump twice) quickly'")
    print("=" * 60)

    rr = RelationalReasoning()

    # First level: jump twice
    jump = rr.create_element("jump", type_tag="action")
    twice = rr.create_element(2, type_tag="quantity_modifier")
    jump_twice = rr.bind(twice, RelationType.MODIFIER, jump)

    print("Level 1: 'jump twice'")
    print(f"  Structure ID: {jump_twice.id}")

    # Second level: modify the whole thing with 'quickly'
    quickly = rr.create_element("quickly", type_tag="manner_modifier")

    # Create a wrapper element for the composed action
    composed_action = rr.create_element(
        content=jump_twice, type_tag="composed_action", element_id="jump_twice_action"
    )

    final = rr.bind(quickly, RelationType.MODIFIER, composed_action)

    print("\nLevel 2: '(jump twice) quickly'")
    print(f"  Final structure ID: {final.id}")
    print(f"  Total elements: {len(final.elements)}")
    print(f"  Total relations: {len(final.relations)}")

    print("✓ Hierarchical composition - structures within structures")
    print()


def demo_schema_extraction():
    """Extract reusable schema from episodes"""
    print("=" * 60)
    print("DEMO 4: Schema Extraction from Episodes")
    print("=" * 60)

    complex = HPCPFCComplex()

    # Encode several similar experiences
    experiences = [
        ("jumped once", {"action": "jump", "count": 1, "success": True}),
        ("ran twice", {"action": "run", "count": 2, "success": True}),
        ("hopped three times", {"action": "hop", "count": 3, "success": True}),
        ("walked once", {"action": "walk", "count": 1, "success": True}),
    ]

    print("Encoding experiences:")
    for content, context in experiences:
        ep, schema = complex.encode_and_abstract(content, context)
        print(f"  - {content}")

    # Check if schema was extracted
    print(f"\nSchemas extracted: {len(complex.pfc.schemas)}")

    if complex.pfc.schemas:
        for sid, schema in complex.pfc.schemas.items():
            print(f"\nSchema '{sid}':")
            print(f"  Structure: {schema.structure}")
            print(f"  Variable slots: {list(schema.slot_fillers.keys())}")
            print(f"  Confidence: {schema.confidence:.2f}")

    print("✓ Abstract pattern extracted from specific episodes")
    print()


def demo_novel_combination():
    """Use HPC-PFC to create truly novel combinations"""
    print("=" * 60)
    print("DEMO 5: Novel Combination via HPC-PFC")
    print("=" * 60)

    complex = HPCPFCComplex()

    # First, learn about concepts separately
    complex.encode_and_abstract("learned to jump", {"concept": "jump", "skill": True})
    complex.encode_and_abstract("the number two", {"concept": "twice", "quantity": 2})
    complex.encode_and_abstract("backward direction", {"concept": "backward", "direction": True})

    # Now compose novel combination never seen before
    novel = complex.compose_novel(
        concepts=["jump", "twice", "backward"], relation="modified_action"
    )

    print("Known concepts:")
    print("  - jump (a skill)")
    print("  - twice (a quantity)")
    print("  - backward (a direction)")

    print("\nNovel composition: 'jump backward twice'")
    print(f"  Components: {novel['components']}")
    print(f"  Marked as novel: {novel['novel']}")

    print("\n✓ Created combination never experienced before!")
    print("  (This is the power of compositional thinking)")
    print()


def demo_analogical_composition():
    """Use analogy to compose in new domain"""
    print("=" * 60)
    print("DEMO 6: Analogical Composition")
    print("=" * 60)

    rr = RelationalReasoning()

    # Source domain: physical movement
    walk = rr.create_element("walk", type_tag="action", element_id="walk")
    slowly = rr.create_element("slowly", type_tag="modifier", element_id="slowly")
    source = rr.bind(slowly, RelationType.MODIFIER, walk)

    print("Source structure: 'walk slowly'")
    print(f"  {slowly.content} --modifies--> {walk.content}")

    # Target domain: cognitive process
    think = rr.create_element("think", type_tag="action", element_id="think")
    carefully = rr.create_element("carefully", type_tag="modifier", element_id="carefully")

    # Map analogy
    target_elements = {"carefully": carefully, "think": think}
    analogical = rr.analogy(source, target_elements)

    print("\nAnalogical mapping to: 'think carefully'")
    if analogical:
        for rel in analogical.relations:
            src = analogical.elements.get(rel.source)
            tgt = analogical.elements.get(rel.target)
            print(
                f"  {src.content if src else '?'} --[{rel.relation_type.value}]--> "
                f"{tgt.content if tgt else '?'}"
            )
        print(f"\n  (Confidence in analogy: {analogical.relations[0].strength:.2f})")

    print("✓ Relational structure transferred across domains")
    print()


def demo_decomposition():
    """Decompose structure back into parts"""
    print("=" * 60)
    print("DEMO 7: Decomposition - Understanding by Breaking Apart")
    print("=" * 60)

    rr = RelationalReasoning()

    # Build complex structure
    quickly = rr.create_element("quickly", type_tag="modifier")
    eat = rr.create_element("eat", type_tag="action")
    the_apple = rr.create_element("the apple", type_tag="entity")

    eat_apple = rr.bind(eat, RelationType.PATIENT, the_apple)
    full_structure = rr.compose(rr.bind(quickly, RelationType.MODIFIER, eat), eat_apple)

    print("Complex structure: 'quickly eat the apple'")
    print(f"  Total elements: {len(full_structure.elements)}")
    print(f"  Total relations: {len(full_structure.relations)}")

    # Decompose
    elements, relations = rr.decompose(full_structure)

    print("\nDecomposed parts:")
    print("  Elements:")
    for eid, elem in elements.items():
        print(f"    - {elem.content} ({elem.type_tag})")
    print("  Relations:")
    for rel in relations:
        print(f"    - {rel.source} --[{rel.relation_type.value}]--> {rel.target}")

    print("✓ Can break down complex structures into components")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMPOSITIONAL THINKING DEMONSTRATION")
    print("HPC-PFC Complex: Combining Known Concepts in Novel Ways")
    print("=" * 60 + "\n")

    demo_basic_composition()
    demo_action_with_roles()
    demo_nested_composition()
    demo_schema_extraction()
    demo_novel_combination()
    demo_analogical_composition()
    demo_decomposition()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("- Composition: Known concepts can be combined")
    print("- Hierarchy: Structures can contain structures")
    print("- Schemas: Abstract patterns from specific episodes")
    print("- Novelty: Never-seen combinations from known parts")
    print("- Analogy: Transfer structure across domains")
    print("- Decomposition: Break apart to understand")
    print("=" * 60)

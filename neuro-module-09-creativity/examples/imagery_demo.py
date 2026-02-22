"""
Mental Imagery Demo - Multimodal Imagination

Demonstrates the mental imagery system across all modalities:
- Visual imagery (occipital cortex)
- Auditory imagery (temporal cortex)
- Motor imagery (premotor cortex)
- Tactile imagery (somatosensory cortex)

Key 2025 finding: Touch imagery vividness correlates positively
with creative performance through semantic integration.
"""

import sys

sys.path.insert(0, ".")

from src.imagery import VisualImagery, AuditoryImagery, MotorImagery, TactileImagery, ImagerySystem


def main():
    print("=" * 60)
    print("MENTAL IMAGERY DEMONSTRATION")
    print("Multimodal Mental Simulation")
    print("=" * 60)

    # Visual Imagery Demo
    print("\n" + "=" * 60)
    print("VISUAL IMAGERY - Mind's Eye")
    print("Located in occipital cortex")
    print("=" * 60)

    visual = VisualImagery()

    print("\n--- Creating Visual Images ---")

    # Create images
    sunset = visual.visualize("A golden sunset over calm ocean waters")
    print("\nImage: Sunset")
    print(f"  Vividness: {sunset.vividness:.3f}")
    print(f"  Properties: {sunset.properties}")

    cube = visual.create_image(
        "rotating_cube",
        "A red 3D cube floating in space",
        properties={"color": "red", "shape": "cube", "dimension": "3D"},
    )
    print("\nImage: Cube")
    print(f"  Vividness: {cube.vividness:.3f}")

    # Mental rotation
    print("\n--- Mental Rotation ---")
    rotated_45 = visual.rotate(cube.id, 45)
    rotated_90 = visual.rotate(cube.id, 90)
    print(f"  Original cube -> 45° rotation: vividness {rotated_45.vividness:.3f}")
    print(f"  Original cube -> 90° rotation: vividness {rotated_90.vividness:.3f}")

    # Scaling
    print("\n--- Mental Scaling ---")
    scaled_up = visual.scale(cube.id, 2.0)
    scaled_down = visual.scale(cube.id, 0.5)
    print(f"  Scale up 2x: vividness {scaled_up.vividness:.3f}")
    print(f"  Scale down 0.5x: vividness {scaled_down.vividness:.3f}")

    # Morphing
    print("\n--- Image Morphing ---")
    sphere = visual.create_image("sphere", "A blue sphere")
    morphed = visual.morph(cube.id, sphere.id, blend_factor=0.5)
    print(f"  Cube + Sphere morph: {morphed.description}")
    print(f"  Morph vividness: {morphed.vividness:.3f}")

    # Auditory Imagery Demo
    print("\n" + "=" * 60)
    print("AUDITORY IMAGERY - Inner Hearing")
    print("Located in temporal cortex")
    print("=" * 60)

    auditory = AuditoryImagery()

    print("\n--- Inner Speech ---")
    speech = auditory.inner_speech("Hello, this is my inner voice speaking")
    print(f"  Inner speech duration: {speech.duration:.2f}s")
    print(f"  Vividness: {speech.vividness:.3f}")

    print("\n--- Musical Imagination ---")
    melody = auditory.imagine_music("A gentle piano melody in C major", tempo=80)
    print(f"  Music: {melody.description}")
    print(f"  Tempo: {melody.properties.get('tempo', 'unknown')} BPM")
    print(f"  Vividness: {melody.vividness:.3f}")

    print("\n--- Sound Manipulation ---")
    # Pitch change
    higher = auditory.change_pitch(melody.id, semitones=5)
    lower = auditory.change_pitch(melody.id, semitones=-5)
    print(f"  Pitch up: {higher.properties.get('pitch', 'unknown')}")
    print(f"  Pitch down: {lower.properties.get('pitch', 'unknown')}")

    # Tempo change
    faster = auditory.change_tempo(melody.id, factor=1.5)
    slower = auditory.change_tempo(melody.id, factor=0.5)
    print(f"  Faster (1.5x): duration {faster.duration:.2f}s")
    print(f"  Slower (0.5x): duration {slower.duration:.2f}s")

    print("\n--- Sound Layering ---")
    rhythm = auditory.create_sound("drums", "A steady drum beat", duration=5.0)
    layered = auditory.layer_sounds([melody.id, rhythm.id])
    print(f"  Layered: {layered.description}")
    print(f"  Duration: {layered.duration:.2f}s")

    # Motor Imagery Demo
    print("\n" + "=" * 60)
    print("MOTOR IMAGERY - Action Simulation")
    print("Located in premotor cortex")
    print("=" * 60)

    motor = MotorImagery()

    print("\n--- Action Imagination ---")
    throw = motor.imagine_action(
        "throw_ball",
        "Throwing a ball overhand",
        body_parts=["right arm", "hand", "shoulder"],
        duration=1.0,
    )
    print(f"  Action: {throw.action}")
    print(f"  Body parts: {throw.body_parts}")
    print(f"  Duration: {throw.duration:.2f}s")
    print(f"  Kinesthetic feel: {throw.kinesthetic_feel:.3f}")

    print("\n--- Mental Rehearsal ---")
    initial_vividness = throw.vividness
    initial_feel = throw.kinesthetic_feel
    motor.rehearse(throw.id, repetitions=5)
    print(f"  Before rehearsal: vividness={initial_vividness:.3f}, feel={initial_feel:.3f}")
    print(
        f"  After 5 repetitions: vividness={throw.vividness:.3f}, feel={throw.kinesthetic_feel:.3f}"
    )

    print("\n--- Slow Motion ---")
    slow_throw = motor.slow_motion(throw.id, factor=0.25)
    print(f"  Original duration: {throw.duration:.2f}s")
    print(f"  Slow motion (0.25x): {slow_throw.duration:.2f}s")

    print("\n--- Action Sequencing ---")
    step = motor.imagine_action("step", "Take a step forward", duration=0.5)
    turn = motor.imagine_action("turn", "Turn 90 degrees", duration=0.3)
    jump = motor.imagine_action("jump", "Jump up", duration=0.4)

    sequence = motor.sequence_actions([step.id, turn.id, jump.id])
    print(f"  Sequence: {sequence.action}")
    print(f"  Total duration: {sequence.duration:.2f}s")

    # Tactile Imagery Demo
    print("\n" + "=" * 60)
    print("TACTILE IMAGERY - Touch Simulation")
    print("Located in somatosensory cortex")
    print("=" * 60)

    tactile = TactileImagery()

    print("\n--- Texture Imagination ---")
    textures = [
        "soft fluffy blanket",
        "rough sandpaper",
        "smooth glass surface",
        "warm sunshine on skin",
    ]

    for texture_desc in textures:
        texture = tactile.feel_texture(texture_desc)
        print(f"\n  Texture: {texture_desc}")
        print(f"    Properties: {texture.properties}")
        print(f"    Semantic associations: {texture.semantic_associations}")

    print("\n--- KEY FINDING: Touch-Creativity Connection ---")
    print("  2025 research shows touch imagery facilitates creative writing")
    print("  through semantic integration and reorganization.")

    # Demonstrate semantic richness
    silk = tactile.feel_texture("smooth cool silk")
    fire = tactile.feel_texture("hot burning fire")
    ice = tactile.feel_texture("cold wet ice")

    print("\n  Semantic associations from touch:")
    print(f"    Silk: {silk.semantic_associations}")
    print(f"    Fire: {fire.semantic_associations}")
    print(f"    Ice: {ice.semantic_associations}")

    print("\n--- Enhancing Abstract Concepts with Touch ---")
    abstract_concepts = ["love", "fear", "excitement", "peace"]

    for concept in abstract_concepts:
        touch = tactile.enhance_with_touch(concept)
        print(f"\n  {concept.upper()}:")
        print(f"    Tactile grounding: {touch.description}")
        print(f"    Associations: {touch.semantic_associations}")

    # Integrated Imagery System Demo
    print("\n" + "=" * 60)
    print("INTEGRATED IMAGERY SYSTEM")
    print("Multimodal Mental Simulation")
    print("=" * 60)

    imagery = ImagerySystem()

    print("\n--- Creating Multimodal Images ---")

    # Full scene imagination
    beach_scene = imagery.imagine_scene("A tropical beach at sunset")
    print("\nScene: Tropical Beach")
    print(f"  Overall vividness: {beach_scene.overall_vividness:.3f}")
    print(f"  Coherence: {beach_scene.coherence:.3f}")
    print(f"  Visual: {'Present' if beach_scene.visual else 'Absent'}")
    print(f"  Auditory: {'Present' if beach_scene.auditory else 'Absent'}")
    print(f"  Tactile: {'Present' if beach_scene.tactile else 'Absent'}")

    # Action imagination
    swimming = imagery.imagine_action("Swimming in the ocean")
    print("\nAction: Swimming")
    print(f"  Overall vividness: {swimming.overall_vividness:.3f}")
    print(f"  Visual: {'Present' if swimming.visual else 'Absent'}")
    print(f"  Motor: {'Present' if swimming.motor else 'Absent'}")
    print(f"  Tactile: {'Present' if swimming.tactile else 'Absent'}")

    # Custom modality selection
    custom = imagery.simulate_experience(
        "Playing a violin", modalities=["visual", "motor", "auditory"]
    )
    print("\nExperience: Violin Playing")
    print(f"  Visual: {'Present' if custom.visual else 'Absent'}")
    print(f"  Motor: {'Present' if custom.motor else 'Absent'}")
    print(f"  Auditory: {'Present' if custom.auditory else 'Absent'}")
    print(f"  Tactile: {'Present (bonus)' if custom.tactile else 'Absent'}")

    print("\n--- Semantic Associations from Multimodal Image ---")
    rich_image = imagery.create_multimodal_image(
        "cozy_fireplace",
        "A warm cozy fireplace with crackling fire",
        include_visual=True,
        include_auditory=True,
        include_tactile=True,
    )

    associations = imagery.get_all_associations(rich_image.id)
    print("\n  Associations from 'cozy fireplace':")
    print(f"    {associations}")

    print("\n--- Image Transformation ---")
    rotated_scene = imagery.transform_multimodal(beach_scene.id, "rotate")
    print(f"\n  Transformed scene: {rotated_scene.description}")
    print(f"  Vividness after rotation: {rotated_scene.overall_vividness:.3f}")

    print("\n--- Image Blending ---")
    forest = imagery.imagine_scene("A misty forest")
    blend = imagery.blend_images(beach_scene.id, forest.id, blend_factor=0.5)
    print(f"\n  Blended: {blend.description}")
    print(f"  Blend vividness: {blend.overall_vividness:.3f}")
    print(f"  Blend coherence: {blend.coherence:.3f}")

    print("\n--- Vividness by Modality ---")
    vividness = imagery.get_vividness_by_modality(beach_scene.id)
    print("\n  Beach scene vividness breakdown:")
    for modality, score in vividness.items():
        print(f"    {modality}: {score:.3f}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY: IMAGERY AND CREATIVITY")
    print("=" * 60)

    print("""
  Mental imagery enables:
  1. Visual simulation - seeing possibilities
  2. Auditory simulation - hearing ideas
  3. Motor simulation - feeling actions
  4. Tactile simulation - grounding in physical experience

  KEY INSIGHT: Tactile imagery provides semantic associations that
  enhance creative thinking by connecting abstract concepts to
  concrete physical experiences.

  Rich multimodal imagery = More creative possibilities
""")


if __name__ == "__main__":
    main()

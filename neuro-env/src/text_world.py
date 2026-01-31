"""
Text World: Text-based adventure environment.

Provides language-grounded learning through text descriptions
and natural language commands.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import re

from base_env import NeuroEnvironment, EnvironmentConfig, StepResult


@dataclass
class Item:
    """An item in the text world."""
    name: str
    description: str
    portable: bool = True
    value: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Room:
    """A room in the text world."""
    name: str
    description: str
    exits: Dict[str, str] = field(default_factory=dict)  # direction -> room_name
    items: List[Item] = field(default_factory=list)
    visited: bool = False


@dataclass
class TextWorldConfig(EnvironmentConfig):
    """Configuration for text world."""
    vocab_size: int = 1000
    max_description_length: int = 100
    reward_item_pickup: float = 0.1
    reward_new_room: float = 0.2
    reward_goal: float = 1.0


class TextWorld(NeuroEnvironment):
    """
    A text-based adventure environment.

    The agent receives text descriptions as observations
    and must output actions that map to commands.

    This bridges language and action for the cognitive system.
    """

    # Basic vocabulary for encoding/decoding
    ACTIONS = ["north", "south", "east", "west", "take", "drop", "look", "inventory"]

    def __init__(
        self,
        config: Optional[TextWorldConfig] = None,
        rooms: Optional[Dict[str, Room]] = None,
    ):
        self.text_config = config or TextWorldConfig()
        super().__init__(self.text_config)

        # Build world
        self._rooms: Dict[str, Room] = rooms or self._create_default_world()
        self._current_room: str = "start"
        self._inventory: List[Item] = []
        self._goal_items: Set[str] = {"treasure", "key"}
        self._collected_goals: Set[str] = set()
        self._last_description: str = ""

        # Build vocabulary
        self._build_vocabulary()

    def _create_default_world(self) -> Dict[str, Room]:
        """Create a simple default world."""
        return {
            "start": Room(
                name="start",
                description="You are in a small room. There is a door to the north.",
                exits={"north": "hallway"},
                items=[Item("torch", "A burning torch", value=0.05)],
            ),
            "hallway": Room(
                name="hallway",
                description="A long hallway stretches before you. Doors lead north, south, and east.",
                exits={"north": "treasury", "south": "start", "east": "garden"},
                items=[],
            ),
            "treasury": Room(
                name="treasury",
                description="A grand treasury! Gold glitters everywhere.",
                exits={"south": "hallway"},
                items=[Item("treasure", "A golden treasure chest", value=1.0)],
            ),
            "garden": Room(
                name="garden",
                description="A peaceful garden with flowers and a fountain.",
                exits={"west": "hallway", "north": "tower"},
                items=[Item("flower", "A beautiful flower", value=0.1)],
            ),
            "tower": Room(
                name="tower",
                description="The top of a tall tower. You can see far and wide.",
                exits={"south": "garden"},
                items=[Item("key", "An ornate key", value=0.5)],
            ),
        }

    def _build_vocabulary(self) -> None:
        """Build vocabulary from world content."""
        words = set()

        # Add action words
        words.update(self.ACTIONS)

        # Add words from rooms
        for room in self._rooms.values():
            words.update(room.name.lower().split())
            words.update(room.description.lower().split())
            for item in room.items:
                words.update(item.name.lower().split())
                words.update(item.description.lower().split())

        # Common words
        words.update(["the", "a", "is", "are", "you", "to", "in", "on", "and", "your"])

        self._vocab = {word: idx for idx, word in enumerate(sorted(words))}
        self._idx_to_word = {idx: word for word, idx in self._vocab.items()}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._episode_count += 1
        self._current_room = "start"
        self._inventory = []
        self._collected_goals = set()

        # Reset room visited status
        for room in self._rooms.values():
            room.visited = False

        self._rooms[self._current_room].visited = True
        self._last_description = self._get_description()
        obs = self._encode_description(self._last_description)

        return obs, {"description": self._last_description}

    def step(self, action: np.ndarray) -> StepResult:
        action = self._normalize_action(action)
        self._step_count += 1

        # Decode action vector to command
        command = self._decode_action(action)
        result, reward = self._execute_command(command)

        self._total_reward += reward
        self._last_description = result
        obs = self._encode_description(result)

        # Check for goal completion
        terminated = self._collected_goals == self._goal_items
        truncated = self._step_count >= self.config.max_steps

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "command": command,
                "description": result,
                "inventory": [i.name for i in self._inventory],
                "room": self._current_room,
            },
        )

    def render(self) -> Optional[str]:
        return self._last_description

    def _get_description(self) -> str:
        """Get current room description."""
        room = self._rooms[self._current_room]
        desc = room.description

        if room.items:
            item_names = ", ".join(i.name for i in room.items)
            desc += f" You see: {item_names}."

        exits = ", ".join(room.exits.keys())
        desc += f" Exits: {exits}."

        return desc

    def _execute_command(self, command: str) -> Tuple[str, float]:
        """Execute a text command and return result and reward."""
        command = command.lower().strip()
        parts = command.split()

        if not parts:
            return "I don't understand.", -0.01

        action = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        reward = 0.0

        room = self._rooms[self._current_room]

        # Movement commands
        if action in ["north", "south", "east", "west", "n", "s", "e", "w"]:
            direction = {"n": "north", "s": "south", "e": "east", "w": "west"}.get(action, action)

            if direction in room.exits:
                self._current_room = room.exits[direction]
                new_room = self._rooms[self._current_room]

                if not new_room.visited:
                    new_room.visited = True
                    reward = self.text_config.reward_new_room

                return self._get_description(), reward
            else:
                return "You can't go that way.", -0.01

        # Take item
        elif action == "take" and args:
            item_name = " ".join(args)
            for item in room.items:
                if item.name.lower() == item_name:
                    if item.portable:
                        room.items.remove(item)
                        self._inventory.append(item)
                        reward = self.text_config.reward_item_pickup + item.value

                        if item.name in self._goal_items:
                            self._collected_goals.add(item.name)
                            reward = self.text_config.reward_goal

                        return f"You take the {item.name}.", reward
                    else:
                        return f"You can't take the {item.name}.", -0.01
            return f"There is no {item_name} here.", -0.01

        # Drop item
        elif action == "drop" and args:
            item_name = " ".join(args)
            for item in self._inventory:
                if item.name.lower() == item_name:
                    self._inventory.remove(item)
                    room.items.append(item)
                    return f"You drop the {item.name}.", 0.0
            return f"You don't have a {item_name}.", -0.01

        # Look
        elif action == "look":
            return self._get_description(), 0.0

        # Inventory
        elif action in ["inventory", "inv", "i"]:
            if self._inventory:
                items = ", ".join(i.name for i in self._inventory)
                return f"You are carrying: {items}.", 0.0
            else:
                return "You are not carrying anything.", 0.0

        else:
            return "I don't understand that command.", -0.01

    def _encode_description(self, text: str) -> np.ndarray:
        """Encode text description to observation vector."""
        obs = np.zeros(self.config.observation_dim, dtype=np.float32)

        # Simple character-level encoding
        words = text.lower().split()[:self.text_config.max_description_length]

        for i, word in enumerate(words):
            # Clean word
            word = re.sub(r'[^\w]', '', word)
            if word in self._vocab:
                idx = self._vocab[word]
                obs_idx = i % self.config.observation_dim
                obs[obs_idx] += (idx + 1) / len(self._vocab)

        # Normalize
        norm = np.linalg.norm(obs)
        if norm > 0:
            obs = obs / norm

        return obs

    def _decode_action(self, action: np.ndarray) -> str:
        """Decode action vector to text command."""
        # Use first few dimensions to select action type
        action_scores = action[:len(self.ACTIONS)]
        action_idx = int(np.argmax(action_scores))
        base_action = self.ACTIONS[action_idx]

        # For take/drop, use remaining dimensions to select object
        if base_action in ["take", "drop"]:
            room = self._rooms[self._current_room]
            items = room.items if base_action == "take" else self._inventory

            if items:
                item_scores = action[len(self.ACTIONS):len(self.ACTIONS) + len(items)]
                if len(item_scores) > 0 and len(items) > 0:
                    item_idx = int(np.argmax(item_scores)) % len(items)
                    return f"{base_action} {items[item_idx].name}"

            return base_action

        return base_action


class ProcGenTextWorld(TextWorld):
    """Procedurally generated text world for varied experiences."""

    def __init__(
        self,
        n_rooms: int = 10,
        config: Optional[TextWorldConfig] = None,
    ):
        self.n_rooms = n_rooms
        # Generate world before calling super().__init__
        rooms = self._generate_world(n_rooms)
        super().__init__(config=config, rooms=rooms)
        # Set current room to first generated room
        self._start_room = list(self._rooms.keys())[0]
        self._current_room = self._start_room

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._episode_count += 1
        self._current_room = self._start_room
        self._inventory = []
        self._collected_goals = set()

        # Reset room visited status
        for room in self._rooms.values():
            room.visited = False

        self._rooms[self._current_room].visited = True
        self._last_description = self._get_description()
        obs = self._encode_description(self._last_description)

        return obs, {"description": self._last_description}

    def _generate_world(self, n_rooms: int) -> Dict[str, Room]:
        """Procedurally generate a world."""
        rng = np.random.default_rng()

        room_types = ["chamber", "hall", "room", "corridor", "cave", "garden", "tower"]
        adjectives = ["dark", "bright", "ancient", "mysterious", "grand", "small", "dusty"]
        item_types = ["key", "gem", "scroll", "coin", "torch", "book", "potion"]

        rooms = {}
        room_names = [f"room_{i}" for i in range(n_rooms)]

        for i, name in enumerate(room_names):
            room_type = rng.choice(room_types)
            adj = rng.choice(adjectives)

            # Generate exits (ensure connected graph)
            exits = {}
            if i > 0:
                exits["back"] = room_names[i - 1]
            if i < n_rooms - 1:
                exits["forward"] = room_names[i + 1]
            if i > 1 and rng.random() < 0.3:
                exits["shortcut"] = room_names[rng.integers(0, i)]

            # Generate items
            items = []
            if rng.random() < 0.5:
                item_type = rng.choice(item_types)
                items.append(Item(
                    name=item_type,
                    description=f"A {adj} {item_type}",
                    value=float(rng.random()),
                ))

            rooms[name] = Room(
                name=name,
                description=f"You are in a {adj} {room_type}.",
                exits=exits,
                items=items,
            )

        # Add goal items to last rooms
        if n_rooms > 0:
            rooms[room_names[-1]].items.append(
                Item("treasure", "The legendary treasure!", value=1.0)
            )

        return rooms

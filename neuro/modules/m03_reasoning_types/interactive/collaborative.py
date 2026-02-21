"""
Collaborative Reasoning - Multi-Agent Problem Solving

Coordinate with other agents through information sharing,
goal negotiation, task division, and solution integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class MessageType(Enum):
    INFORMATION = "information"
    REQUEST = "request"
    PROPOSAL = "proposal"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    ACKNOWLEDGMENT = "acknowledgment"


@dataclass
class Message:
    """A message between agents"""
    message_id: str
    sender: str
    recipient: str
    msg_type: MessageType
    content: Any
    timestamp: float = 0.0
    in_reply_to: Optional[str] = None


@dataclass
class Task:
    """A task that can be divided and assigned"""
    task_id: str
    description: str
    requirements: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    assigned_to: Optional[str] = None
    completed: bool = False
    result: Any = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Agent:
    """Representation of a collaborating agent"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    current_task: Optional[str] = None
    knowledge: Dict[str, Any] = field(default_factory=dict)
    available: bool = True


@dataclass
class SharedGoal:
    """A goal shared among multiple agents"""
    goal_id: str
    description: str
    participants: Set[str] = field(default_factory=set)
    agreed: bool = False
    progress: float = 0.0
    tasks: List[str] = field(default_factory=list)


class JointAttention:
    """Shared focus mechanism between agents"""

    def __init__(self):
        self.attention_targets: Dict[str, str] = {}
        self.joint_focus: Optional[str] = None
        self.attention_history: List[Tuple[str, str, float]] = []

    def set_attention(self, agent_id: str, target: str, timestamp: float = 0.0):
        """Set what an agent is attending to"""
        self.attention_targets[agent_id] = target
        self.attention_history.append((agent_id, target, timestamp))
        self._check_joint_attention()

    def _check_joint_attention(self):
        """Check if multiple agents share focus"""
        if not self.attention_targets:
            self.joint_focus = None
            return

        targets = list(self.attention_targets.values())
        if len(set(targets)) == 1 and len(targets) > 1:
            self.joint_focus = targets[0]
        else:
            self.joint_focus = None

    def establish_joint_attention(self,
                                   initiator: str,
                                   target: str,
                                   other_agents: List[str]
                                   ) -> bool:
        """Try to establish joint attention on a target"""
        self.set_attention(initiator, target)

        for agent in other_agents:
            self.set_attention(agent, target)

        return self.joint_focus == target

    def follow_gaze(self, follower: str, leader: str) -> Optional[str]:
        """Have follower look where leader is looking"""
        if leader in self.attention_targets:
            target = self.attention_targets[leader]
            self.set_attention(follower, target)
            return target
        return None

    def point_to(self, pointer: str, target: str, observers: List[str]):
        """Direct others' attention to a target"""
        self.set_attention(pointer, target)
        for observer in observers:
            self.set_attention(observer, target)


class CollaborativeReasoner:
    """
    Coordinate with other agents for multi-agent problem solving.

    Capabilities:
    - Share information between agents
    - Negotiate shared goals
    - Divide tasks based on capabilities
    - Merge partial solutions
    """

    def __init__(self, self_id: str):
        self.self_id = self_id
        self.agents: Dict[str, Agent] = {}
        self.messages: List[Message] = []
        self.tasks: Dict[str, Task] = {}
        self.shared_goals: Dict[str, SharedGoal] = {}
        self.joint_attention = JointAttention()
        self.shared_knowledge: Dict[str, Any] = {}

        self.agents[self_id] = Agent(agent_id=self_id)

    def register_agent(self, agent: Agent):
        """Register a collaborating agent"""
        self.agents[agent.agent_id] = agent

    def share_information(self,
                          info: Dict[str, Any],
                          recipients: List[str] = None
                          ) -> List[Message]:
        """Share information with other agents"""
        if recipients is None:
            recipients = [a for a in self.agents.keys() if a != self.self_id]

        messages = []
        for recipient in recipients:
            msg = Message(
                message_id=f"msg_{len(self.messages)}",
                sender=self.self_id,
                recipient=recipient,
                msg_type=MessageType.INFORMATION,
                content=info,
                timestamp=len(self.messages)
            )
            self.messages.append(msg)
            messages.append(msg)

            if recipient in self.agents:
                self.agents[recipient].knowledge.update(info)

        self.shared_knowledge.update(info)
        return messages

    def request_information(self,
                            query: str,
                            from_agent: str
                            ) -> Message:
        """Request information from another agent"""
        msg = Message(
            message_id=f"msg_{len(self.messages)}",
            sender=self.self_id,
            recipient=from_agent,
            msg_type=MessageType.REQUEST,
            content={"query": query},
            timestamp=len(self.messages)
        )
        self.messages.append(msg)
        return msg

    def respond_to_request(self,
                           request_msg: Message,
                           response: Any
                           ) -> Message:
        """Respond to an information request"""
        msg = Message(
            message_id=f"msg_{len(self.messages)}",
            sender=self.self_id,
            recipient=request_msg.sender,
            msg_type=MessageType.INFORMATION,
            content=response,
            timestamp=len(self.messages),
            in_reply_to=request_msg.message_id
        )
        self.messages.append(msg)
        return msg

    def propose_goal(self,
                     goal_description: str,
                     participants: List[str]
                     ) -> SharedGoal:
        """Propose a shared goal to other agents"""
        goal = SharedGoal(
            goal_id=f"goal_{len(self.shared_goals)}",
            description=goal_description,
            participants=set(participants + [self.self_id])
        )
        self.shared_goals[goal.goal_id] = goal

        for participant in participants:
            msg = Message(
                message_id=f"msg_{len(self.messages)}",
                sender=self.self_id,
                recipient=participant,
                msg_type=MessageType.PROPOSAL,
                content={"goal_id": goal.goal_id, "description": goal_description},
                timestamp=len(self.messages)
            )
            self.messages.append(msg)

        return goal

    def negotiate_goal(self,
                       goal_id: str,
                       accept: bool,
                       counter_proposal: str = None
                       ) -> Message:
        """Respond to a goal proposal"""
        if goal_id not in self.shared_goals:
            return None

        goal = self.shared_goals[goal_id]

        if accept:
            goal.agreed = True
            msg_type = MessageType.ACCEPTANCE
            content = {"goal_id": goal_id, "status": "accepted"}
        else:
            msg_type = MessageType.REJECTION
            content = {
                "goal_id": goal_id,
                "status": "rejected",
                "counter_proposal": counter_proposal
            }

        for participant in goal.participants:
            if participant != self.self_id:
                msg = Message(
                    message_id=f"msg_{len(self.messages)}",
                    sender=self.self_id,
                    recipient=participant,
                    msg_type=msg_type,
                    content=content,
                    timestamp=len(self.messages)
                )
                self.messages.append(msg)

        return msg

    def create_task(self,
                    description: str,
                    requirements: List[str] = None,
                    dependencies: List[str] = None
                    ) -> Task:
        """Create a new task"""
        task = Task(
            task_id=f"task_{len(self.tasks)}",
            description=description,
            requirements=requirements or [],
            dependencies=dependencies or []
        )
        self.tasks[task.task_id] = task
        return task

    def divide_task(self,
                    task: Task,
                    available_agents: List[str] = None
                    ) -> Dict[str, List[Task]]:
        """Divide a task among available agents based on capabilities"""
        if available_agents is None:
            available_agents = list(self.agents.keys())

        assignments: Dict[str, List[Task]] = defaultdict(list)

        if not task.subtasks:
            best_agent = self._find_best_agent(task, available_agents)
            if best_agent:
                task.assigned_to = best_agent
                assignments[best_agent].append(task)
            return dict(assignments)

        for subtask in task.subtasks:
            deps_met = all(
                self.tasks.get(dep, Task(task_id=dep, description="")).completed
                for dep in subtask.dependencies
            )

            if deps_met:
                best_agent = self._find_best_agent(subtask, available_agents)
                if best_agent:
                    subtask.assigned_to = best_agent
                    assignments[best_agent].append(subtask)

        return dict(assignments)

    def _find_best_agent(self,
                         task: Task,
                         candidates: List[str]
                         ) -> Optional[str]:
        """Find the best agent for a task based on capabilities"""
        best_agent = None
        best_score = -1

        for agent_id in candidates:
            if agent_id not in self.agents:
                continue

            agent = self.agents[agent_id]
            if not agent.available:
                continue

            matching_caps = len(set(task.requirements) & agent.capabilities)
            total_reqs = len(task.requirements) if task.requirements else 1
            score = matching_caps / total_reqs

            if agent.current_task is None:
                score += 0.5

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent"""
        if task_id not in self.tasks or agent_id not in self.agents:
            return False

        task = self.tasks[task_id]
        agent = self.agents[agent_id]

        task.assigned_to = agent_id
        agent.current_task = task_id

        msg = Message(
            message_id=f"msg_{len(self.messages)}",
            sender=self.self_id,
            recipient=agent_id,
            msg_type=MessageType.INFORMATION,
            content={"task_assignment": task_id, "description": task.description},
            timestamp=len(self.messages)
        )
        self.messages.append(msg)

        return True

    def report_task_completion(self,
                               task_id: str,
                               result: Any
                               ) -> bool:
        """Report completion of a task"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.completed = True
        task.result = result

        if task.assigned_to and task.assigned_to in self.agents:
            self.agents[task.assigned_to].current_task = None

        for goal in self.shared_goals.values():
            if task_id in goal.tasks:
                completed_tasks = sum(
                    1 for tid in goal.tasks
                    if self.tasks.get(tid, Task(task_id=tid, description="")).completed
                )
                goal.progress = completed_tasks / len(goal.tasks)

        return True

    def merge_solutions(self,
                        partial_solutions: List[Tuple[str, Any]],
                        merge_function: Callable[[List[Any]], Any] = None
                        ) -> Any:
        """Merge partial solutions from different agents"""
        if not partial_solutions:
            return None

        if merge_function:
            solutions = [sol for _, sol in partial_solutions]
            return merge_function(solutions)

        if all(isinstance(sol, dict) for _, sol in partial_solutions):
            merged = {}
            for _, sol in partial_solutions:
                merged.update(sol)
            return merged

        if all(isinstance(sol, list) for _, sol in partial_solutions):
            merged = []
            for _, sol in partial_solutions:
                merged.extend(sol)
            return merged

        if all(isinstance(sol, (int, float)) for _, sol in partial_solutions):
            return np.mean([sol for _, sol in partial_solutions])

        return [sol for _, sol in partial_solutions]

    def broadcast(self, content: Any) -> List[Message]:
        """Broadcast a message to all agents"""
        return self.share_information(
            {"broadcast": content},
            recipients=None
        )

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        status = {}
        for agent_id, agent in self.agents.items():
            current_task_info = None
            if agent.current_task and agent.current_task in self.tasks:
                task = self.tasks[agent.current_task]
                current_task_info = {
                    "task_id": task.task_id,
                    "description": task.description
                }

            status[agent_id] = {
                "available": agent.available,
                "current_task": current_task_info,
                "capabilities": list(agent.capabilities),
                "knowledge_keys": list(agent.knowledge.keys())
            }

        return status

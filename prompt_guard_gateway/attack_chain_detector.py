"""Multi-Turn Attack Chain Detection - Track escalation patterns across sessions"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TurnNode:
    """Represents a single turn in conversation."""
    turn_number: int
    text: str
    intent: str
    risk_score: float
    classification: str
    attack_type: str
    timestamp: str


class AttackChainDetector:
    """Detects multi-turn attack chains with gradual escalation."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.sessions = {}  # session_id -> list of TurnNode
    
    def add_turn(
        self,
        session_id: str,
        text: str,
        intent: str,
        risk_score: float,
        classification: str,
        attack_type: str
    ) -> Dict:
        """
        Add a turn and analyze attack chain.
        
        Returns:
            Dict with escalation_detected, escalation_score, attack_graph
        """
        # Initialize session if new
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Create turn node
        turn = TurnNode(
            turn_number=len(self.sessions[session_id]) + 1,
            text=text[:200],  # Truncate for storage
            intent=intent,
            risk_score=risk_score,
            classification=classification,
            attack_type=attack_type,
            timestamp=datetime.now().isoformat()
        )
        
        # Add to session
        self.sessions[session_id].append(turn)
        
        # Keep only recent history
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
        
        # Analyze attack chain
        result = self._analyze_chain(session_id)
        # Always include turn_count, even for single-turn sessions where the
        # chain analysis short-circuits.
        result.setdefault("turn_count", len(self.sessions.get(session_id, [])))
        return result
    
    def _analyze_chain(self, session_id: str) -> Dict:
        """Analyze session for attack chain patterns."""
        turns = self.sessions.get(session_id, [])
        
        if len(turns) < 2:
            return {
                "escalation_detected": False,
                "escalation_score": 0.0,
                "attack_graph": self._build_graph(turns),
                "patterns": []
            }
        
        # Detect patterns
        patterns = []
        
        # 1. Intent Evolution (innocent → malicious)
        intent_evolution = self._detect_intent_evolution(turns)
        if intent_evolution["detected"]:
            patterns.append(intent_evolution)
        
        # 2. Privilege Escalation (user → admin)
        privilege_escalation = self._detect_privilege_escalation(turns)
        if privilege_escalation["detected"]:
            patterns.append(privilege_escalation)
        
        # 3. Semantic Drift (gradual topic shift)
        semantic_drift = self._detect_semantic_drift(turns)
        if semantic_drift["detected"]:
            patterns.append(semantic_drift)
        
        # 4. Risk Escalation (low → high risk)
        risk_escalation = self._detect_risk_escalation(turns)
        if risk_escalation["detected"]:
            patterns.append(risk_escalation)
        
        # Calculate escalation score (exponential if multiple patterns)
        escalation_score = self._calculate_escalation_score(patterns, len(turns))
        
        return {
            "escalation_detected": len(patterns) > 0,
            "escalation_score": escalation_score,
            "attack_graph": self._build_graph(turns),
            "patterns": patterns,
            "turn_count": len(turns)
        }
    
    def _detect_intent_evolution(self, turns: List[TurnNode]) -> Dict:
        """Detect evolution from innocent to malicious intent."""
        if len(turns) < 3:
            return {"detected": False}
        
        # Check if starts innocent and becomes malicious
        recent = turns[-5:]
        
        innocent_start = recent[0].classification in ["SAFE", "REQUIRES_AUTH"]
        malicious_end = recent[-1].classification == "MALICIOUS"
        
        if innocent_start and malicious_end:
            # Check gradual transition
            risk_progression = [t.risk_score for t in recent]
            is_gradual = all(
                risk_progression[i] <= risk_progression[i+1] + 0.2
                for i in range(len(risk_progression)-1)
            )
            
            if is_gradual:
                return {
                    "detected": True,
                    "type": "intent_evolution",
                    "severity": "HIGH",
                    "description": f"Intent evolved from {recent[0].classification} to {recent[-1].classification}",
                    "turns_involved": [t.turn_number for t in recent]
                }
        
        return {"detected": False}
    
    def _detect_privilege_escalation(self, turns: List[TurnNode]) -> Dict:
        """Detect attempts to escalate privileges."""
        privilege_keywords = [
            "admin", "administrator", "root", "sudo", "developer",
            "test", "demo", "debug", "authorized", "special access"
        ]
        
        recent = turns[-5:]
        escalation_turns = []
        
        for turn in recent:
            text_lower = turn.text.lower()
            if any(kw in text_lower for kw in privilege_keywords):
                escalation_turns.append(turn.turn_number)
        
        if len(escalation_turns) >= 2:
            return {
                "detected": True,
                "type": "privilege_escalation",
                "severity": "CRITICAL",
                "description": f"Privilege escalation attempts in {len(escalation_turns)} turns",
                "turns_involved": escalation_turns
            }
        
        return {"detected": False}
    
    def _detect_semantic_drift(self, turns: List[TurnNode]) -> Dict:
        """Detect gradual topic drift toward malicious intent."""
        if len(turns) < 4:
            return {"detected": False}
        
        recent = turns[-6:]
        
        # Track intent changes
        intents = [t.intent for t in recent]
        unique_intents = len(set(intents))
        
        # Drift detected if many intent changes + increasing risk
        if unique_intents >= 3:
            risk_trend = [t.risk_score for t in recent]
            increasing_risk = risk_trend[-1] > risk_trend[0] + 0.3
            
            if increasing_risk:
                return {
                    "detected": True,
                    "type": "semantic_drift",
                    "severity": "MEDIUM",
                    "description": f"Topic drifted across {unique_intents} intents with increasing risk",
                    "turns_involved": [t.turn_number for t in recent]
                }
        
        return {"detected": False}
    
    def _detect_risk_escalation(self, turns: List[TurnNode]) -> Dict:
        """Detect exponential risk increase."""
        if len(turns) < 3:
            return {"detected": False}
        
        recent = turns[-5:]
        risks = [t.risk_score for t in recent]
        
        # Check for consistent increase
        increases = sum(1 for i in range(len(risks)-1) if risks[i+1] > risks[i])
        
        if increases >= len(risks) - 2:  # Most turns show increase
            risk_delta = risks[-1] - risks[0]
            
            if risk_delta > 0.4:
                return {
                    "detected": True,
                    "type": "risk_escalation",
                    "severity": "HIGH",
                    "description": f"Risk increased by {risk_delta:.2f} over {len(recent)} turns",
                    "turns_involved": [t.turn_number for t in recent]
                }
        
        return {"detected": False}
    
    def _calculate_escalation_score(self, patterns: List[Dict], turn_count: int) -> float:
        """Calculate exponential escalation score."""
        if not patterns:
            return 0.0
        
        # Base score from pattern count (exponential)
        base_score = 1.0 - (1.0 / (1.0 + len(patterns) ** 2))
        
        # Severity multiplier
        severity_weights = {"LOW": 1.0, "MEDIUM": 1.5, "HIGH": 2.0, "CRITICAL": 3.0}
        severity_multiplier = sum(
            severity_weights.get(p.get("severity", "MEDIUM"), 1.5)
            for p in patterns
        ) / len(patterns)
        
        # Turn count factor (more turns = more suspicious)
        turn_factor = min(1.0, turn_count / 10.0)
        
        # Exponential combination
        escalation_score = base_score * severity_multiplier * (1.0 + turn_factor)
        
        return min(1.0, escalation_score)
    
    def _build_graph(self, turns: List[TurnNode]) -> Dict:
        """Build attack graph representation."""
        if not turns:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        
        for i, turn in enumerate(turns):
            nodes.append({
                "id": turn.turn_number,
                "text": turn.text[:50] + "..." if len(turn.text) > 50 else turn.text,
                "intent": turn.intent,
                "risk": round(turn.risk_score, 2),
                "classification": turn.classification,
                "attack_type": turn.attack_type
            })
            
            # Create edge to next turn
            if i < len(turns) - 1:
                risk_delta = turns[i+1].risk_score - turn.risk_score
                edges.append({
                    "from": turn.turn_number,
                    "to": turns[i+1].turn_number,
                    "risk_delta": round(risk_delta, 2),
                    "escalation": risk_delta > 0.2
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary of session attack chain."""
        if session_id not in self.sessions:
            return None
        
        turns = self.sessions[session_id]
        analysis = self._analyze_chain(session_id)
        
        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "first_turn": turns[0].timestamp if turns else None,
            "last_turn": turns[-1].timestamp if turns else None,
            "max_risk": max(t.risk_score for t in turns) if turns else 0.0,
            "avg_risk": sum(t.risk_score for t in turns) / len(turns) if turns else 0.0,
            "escalation_detected": analysis["escalation_detected"],
            "escalation_score": analysis["escalation_score"],
            "patterns": analysis["patterns"]
        }
    
    def clear_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Global singleton
_default_detector = None

def get_default_detector() -> AttackChainDetector:
    """Get or create default attack chain detector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = AttackChainDetector()
    return _default_detector

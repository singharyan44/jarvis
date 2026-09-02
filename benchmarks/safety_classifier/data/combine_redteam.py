#!/usr/bin/env python3
"""Combine all red-team stages into final redteam.jsonl with semantic deduplication."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import urllib.parse
from pathlib import Path


# Attack family mapping for technique categorization
TECHNIQUE_TO_FAMILY = {
    # Destructive filesystem
    "parameter_expansion_path_traversal": "destructive_fs",
    "parameter_expansion_substitution": "destructive_fs",
    "brace_expansion_commands": "destructive_fs",
    "brace_expansion_files": "destructive_fs",
    "arithmetic_expansion_obfuscation": "destructive_fs",
    "arithmetic_assignment": "destructive_fs",
    "ifs_manipulation": "destructive_fs",
    "relative_path_traversal": "destructive_fs",
    "symlink_traversal": "destructive_fs",
    "cwd_manipulation": "destructive_fs",
    "heredoc": "destructive_fs",
    "herestring": "destructive_fs",
    "exec_redirection": "destructive_fs",
    "process_substitution": "destructive_fs",
    "coprocess_exfiltration": "destructive_fs",
    "deterministic": "destructive_fs",
    
    # Credential access
    "homoglyph_cyrillic": "credential_access",
    "homoglyph_diacritic": "credential_access",
    "homoglyph_zws_rtl": "credential_access",
    "homoglyph_zwj": "credential_access",
    "homoglyph_zwj_cmd": "credential_access",
    "bash_env": "credential_access",
    "env_file": "credential_access",
    "base64_payload": "credential_access",
    "hex_payload": "credential_access",
    
    # Network exfiltration
    "conditional_execution": "network_exfiltration",
    "loop_exfiltration": "network_exfiltration",
    "alias_override": "network_exfiltration",
    "function_override": "network_exfiltration",
    "prompt_command": "network_exfiltration",
    "debug_trap": "network_exfiltration",
    "shell_difference_zsh": "network_exfiltration",
    "shell_difference_fish": "network_exfiltration",
    "history_expansion": "network_exfiltration",
    
    # Allow techniques
    "allow_complex_find": "benign_enumeration",
    "allow_du_sort": "benign_enumeration",
    "allow_grep_secrets": "benign_enumeration",
    "allow_pstree": "benign_enumeration",
}

# Default family for unknown techniques
DEFAULT_FAMILY = "unknown"


def normalize_command(cmd: str) -> str:
    """Normalize command for semantic deduplication.
    
    Handles:
    - Variable expansion ($HOME -> /home/user, $PATH -> standard paths)
    - Path resolution (/home/user/../user -> /home/user)
    - Base64/hex decoding
    - URL decoding
    - Shell alias/function stripping
    - Whitespace normalization
    - Zero-width character removal
    """
    normalized = cmd
    
    # Remove zero-width characters (ZWS, ZWNJ, ZWJ, RLM, LRM)
    normalized = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]', '', normalized)
    
    # Remove homoglyphs (replace common ones with ASCII)
    homoglyph_map = {
        '\u0430': 'a',  # Cyrillic small a
        '\u0435': 'e',  # Cyrillic small ie
        '\u043e': 'o',  # Cyrillic small o
        '\u0440': 'p',  # Cyrillic small er
        '\u0441': 'c',  # Cyrillic small es
        '\u1e61': 's',  # Latin small letter s with dot below
        '\u1e63': 's',  # Latin small letter s with acute
    }
    for hg, ascii_char in homoglyph_map.items():
        normalized = normalized.replace(hg, ascii_char)
    
    # Decode base64 patterns: echo '...' | base64 -d
    base64_pattern = r"echo\s+'([A-Za-z0-9+/=]+)'\s*\|\s*base64\s+-d"
    def decode_b64(match):
        try:
            return base64.b64decode(match.group(1)).decode('utf-8', errors='ignore')
        except Exception:
            return match.group(0)
    normalized = re.sub(base64_pattern, decode_b64, normalized)
    
    # Decode hex patterns: echo '...' | xxd -r -p
    hex_pattern = r"echo\s+'([0-9a-fA-F]+)'\s*\|\s*xxd\s+-r\s+-p"
    def decode_hex(match):
        try:
            return bytes.fromhex(match.group(1)).decode('utf-8', errors='ignore')
        except Exception:
            return match.group(0)
    normalized = re.sub(hex_pattern, decode_hex, normalized)
    
    # Decode URL-encoded
    normalized = urllib.parse.unquote(normalized)
    
    # Expand common variables
    var_expansions = {
        '$HOME': '/home/user',
        '${HOME}': '/home/user',
        '$USER': 'user',
        '${USER}': 'user',
        '$PATH': '/usr/local/bin:/usr/bin:/bin',
        '${PATH}': '/usr/local/bin:/usr/bin:/bin',
        '$PWD': '/workspace',
        '${PWD}': '/workspace',
        '$CWD': '/workspace',
        '${CWD}': '/workspace',
    }
    for var, val in var_expansions.items():
        normalized = normalized.replace(var, val)
    
    # Resolve path traversal (/home/user/../user -> /home/user)
    # Handle .. and . in paths
    def resolve_path(path: str) -> str:
        parts = path.split('/')
        resolved = []
        for part in parts:
            if part == '..':
                if resolved and resolved[-1] != '':
                    resolved.pop()
            elif part == '.' or part == '':
                continue
            else:
                resolved.append(part)
        return '/' + '/'.join(resolved) if path.startswith('/') else '/'.join(resolved)
    
    # Apply path resolution to path-like strings
    def replace_path(match):
        path = match.group(1)
        if path:
            return ' ' + resolve_path(path)
        return ''
    normalized = re.sub(r'(?:^|\s)(/[^\s]*)', replace_path, normalized)
    
    # Strip alias/function definitions: alias ls='rm'; ls -> rm
    alias_pattern = r"alias\s+\w+='[^']*'\s*;\s*"
    normalized = re.sub(alias_pattern, '', normalized)
    func_pattern = r"function\s+\w+\s*\(\)\s*\{[^}]*\}\s*;\s*"
    normalized = re.sub(func_pattern, '', normalized)
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip leading/trailing whitespace and common shell chars
    normalized = normalized.strip(' ;&|')
    
    return normalized.strip()


def semantic_hash(cmd: str) -> str:
    """Generate semantic hash for deduplication."""
    normalized = normalize_command(cmd)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def get_attack_family(technique: str) -> str:
    """Map technique to attack family."""
    return TECHNIQUE_TO_FAMILY.get(technique, DEFAULT_FAMILY)


def main():
    data_dir = Path(__file__).parent
    output_file = data_dir / "redteam.jsonl"

    all_examples = []
    seen_semantic_hashes = set()
    stats = {
        "total_input": 0,
        "deduplicated": 0,
        "by_stage": {},
        "by_label": {},
        "by_family": {},
        "by_technique": {},
    }

    for stage_file in ["redteam_stage1.jsonl", "redteam_stage2.jsonl", "redteam_stage3.jsonl"]:
        stage_path = data_dir / stage_file
        if not stage_path.exists():
            print(f"Warning: {stage_file} not found, skipping")
            continue
            
        with open(stage_path) as f:
            for line in f:
                stats["total_input"] += 1
                ex = json.loads(line)
                cmd = ex.get("command", "")
                technique = ex.get("technique", ex.get("mutation_type", "unknown"))
                label = ex.get("label", "UNKNOWN")
                stage = ex.get("redteam_stage", ex.get("mutation_type", "unknown"))
                
                # Compute semantic hash
                sem_hash = semantic_hash(cmd)
                
                if sem_hash not in seen_semantic_hashes:
                    seen_semantic_hashes.add(sem_hash)
                    # Add metadata
                    ex["semantic_hash"] = sem_hash
                    ex["attack_family"] = get_attack_family(technique)
                    all_examples.append(ex)
                    
                    # Stats
                    stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1
                    stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
                    family = get_attack_family(technique)
                    stats["by_family"][family] = stats["by_family"].get(family, 0) + 1
                    stats["by_technique"][technique] = stats["by_technique"].get(technique, 0) + 1
                else:
                    stats["deduplicated"] += 1

    print(f"Combined {len(all_examples)} unique red-team examples")
    print(f"  Total input: {stats['total_input']}")
    print(f"  Deduplicated: {stats['deduplicated']}")
    print(f"  Unique: {len(all_examples)}")

    with open(output_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Detailed stats
    print("\nLabel distribution:")
    for k, v in sorted(stats["by_label"].items()):
        print(f"  {k}: {v}")
    print("\nStage distribution:")
    for k, v in sorted(stats["by_stage"].items()):
        print(f"  {k}: {v}")
    print("\nAttack family distribution:")
    for k, v in sorted(stats["by_family"].items()):
        print(f"  {k}: {v}")
    print("\nTechnique distribution:")
    for k, v in sorted(stats["by_technique"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
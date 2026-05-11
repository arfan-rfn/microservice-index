"""Scenario generator for AI test project authorization flow testing.

This module generates authorization scenarios from all_vectors.json by analyzing
permission matrices and call chains to detect potential inconsistencies in
authorization policies across microservice communication paths.

The generator focuses on:
- Extracting permissions by endpoint from permission_matrix
- Classifying scenarios as EntryPointAuthorization or DownstreamPolicyConsistency
- Detecting policy inconsistencies along call chains

Downstream Authorization Policy Consistency Testing:
This testing phase verifies the consistency of authorization policies across
inter-service communication paths. It analyzes whether roles and permissions
required by downstream services are coherent with upstream entry points.

Examples of inconsistencies detected:
- Over-permission: Upstream requires 'admin' but downstream allows 'user'
- Under-permission: Upstream allows 'user' but downstream requires 'admin'

Goal:
Ensure all downstream calls preserve intended authorization boundaries,
preventing privilege escalation or inconsistent access control.
"""

import json
import argparse
from collections import Counter
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ir_feature_extractor import analyze_ir_node

StatusMap = {1: "2xx", 0: "403", -1: "UNKNOWN"}

# =====================================================
#  LOAD AUXILIARY FILES
# =====================================================
def load_json(path: str):
    """Loads and normalizes JSON data from a file.
    
    This function loads JSON data from the specified path and normalizes it
    to a list format. If the data is a dictionary with 'endpoints' or 
    'components' keys, it extracts the values into a list.
    
    Args:
        path (str): The file path to the JSON file.
        
    Returns:
        list: The normalized JSON data as a list.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # normalize if dict
    if isinstance(data, dict):
        # case: top-level "endpoints" or "components"
        if "endpoints" in data and isinstance(data["endpoints"], dict):
            data = list(data["endpoints"].values())
        elif "components" in data and isinstance(data["components"], dict):
            data = list(data["components"].values())

    return data


# =====================================================
#  SCENARIO CLASSIFICATION
# =====================================================
def classify_scenario(call_chain: List[Dict[str, Any]]) -> str:
    """Classifies a scenario based on its call chain characteristics.
    
    Determines whether a scenario is an EntryPointAuthorization (single endpoint)
    or DownstreamPolicyConsistency (multiple endpoints with depth > 0) scenario.
    
    Args:
        call_chain (List[Dict[str, Any]]): A list of dictionaries representing
            the call chain, where each dictionary contains endpoint information
            and depth information.
            
    Returns:
        str: Either "DownstreamPolicyConsistency" if any call in the chain
            has depth > 0, or "EntryPointAuthorization" otherwise.
    """
    return "DownstreamPolicyConsistency" if any(c.get("depth", 0) > 0 for c in call_chain) else "EntryPointAuthorization"


# =====================================================
#  PERMISSION EXTRACTION
# =====================================================
def _row_for_endpoint(pm: Dict[str, Any], endpoint_id: str) -> Tuple[List[int], List[str]]:
    """Extracts the permission row and roles for a specific endpoint from the permission matrix.
    
    Args:
        pm (Dict[str, Any]): The permission matrix dictionary containing 'roles',
            'endpoints', and 'matrix' keys.
        endpoint_id (str): The ID of the endpoint to look up in the matrix.
        
    Returns:
        Tuple[List[int], List[str]]: A tuple containing:
            - A list of integers representing the permission values for the endpoint
            - A list of role names corresponding to the permission matrix columns
            
    Returns empty lists if the endpoint is not found or the matrix is invalid.
    """
    roles: List[str] = pm.get("roles", [])
    endpoints: List[str] = pm.get("endpoints", [])
    matrix: List[List[int]] = pm.get("matrix", [])
    if not endpoints or not matrix:
        return [], roles
    try:
        row_idx = endpoints.index(endpoint_id)
    except ValueError:
        return [], roles
    return matrix[row_idx], roles


def resolve_permission_matrix_endpoint(pm: Dict[str, Any], chain_node: Dict[str, Any]) -> "str | None":
    endpoints: List[str] = pm.get("endpoints", [])
    endpoint_id = chain_node.get("endpoint_id")
    uri = chain_node.get("uri")

    if endpoint_id and endpoint_id in endpoints:
        return endpoint_id

    if uri:
        for candidate in endpoints:
            if candidate == uri:
                return candidate
            if candidate.startswith("UNRESOLVED:") and candidate[len("UNRESOLVED:"):] == uri:
                return candidate
            if candidate.endswith(uri):
                return candidate

    return None


def split_permission_row(roles: List[str], row: List[int], is_public_hint: bool = False) -> Dict[str, Any]:
    """Split a permission matrix row into allowed/denied/unknown role buckets.

    When *is_public_hint* is True (caller knows the endpoint is genuinely public,
    e.g. no authentication required) all -1 values are promoted to allowed (2xx).
    Without the hint every -1 is treated as unknown — this is the correct behaviour
    for UNRESOLVED downstream nodes where we simply have no permission data.
    """
    allowed, denied, unknown = [], [], []
    expected_status_by_role = {}
    for i, v in enumerate(row):
        role = roles[i]
        if v == 1 or (v == -1 and is_public_hint):
            allowed.append(role)
            expected_status_by_role[role] = "2xx"
        elif v == 0:
            denied.append(role)
            expected_status_by_role[role] = "403"
        else:
            unknown.append(role)
            expected_status_by_role[role] = "UNKNOWN"

    is_public_endpoint = bool(allowed) and not denied and not unknown

    return {
        "allowed_roles": allowed,
        "denied_roles": denied,
        "unknown_roles": unknown,
        "expected_status_by_role": expected_status_by_role,
        "is_public_endpoint": is_public_endpoint,
    }


def collect_chain_permission_rows(pm: Dict[str, Any], call_chain: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    roles: List[str] = pm.get("roles", [])

    for chain_node in call_chain:
        resolved_endpoint = resolve_permission_matrix_endpoint(pm, chain_node)
        if not resolved_endpoint:
            continue

        row, row_roles = _row_for_endpoint(pm, resolved_endpoint)
        if not row or not row_roles:
            continue

        # Nodes that could not be resolved at static analysis time (resolved=False,
        # endpoint_id=None) have permission values of -1, but -1 here means
        # "unresolved/unknown" — NOT "publicly accessible".  Force the values to -1
        # so that split_permission_row places roles in unknown_roles rather than
        # treating the endpoint as public.
        is_unresolved = not chain_node.get("resolved", True) or chain_node.get("endpoint_id") is None
        effective_values = [-1] * len(row) if is_unresolved else row

        rows.append({
            "endpoint": resolved_endpoint,
            "values": effective_values,
            "roles": row_roles,
            "is_unresolved": is_unresolved,
        })

    return roles, rows


UNRESOLVED_ENDPOINT_PREFIX = "UNRESOLVED:"


def partition_chain_rows_by_resolution(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition ordered chain rows into a resolved prefix and an unresolved
    black box.

    Once an unresolved endpoint is reached, every subsequent node is folded
    into the same unresolved block — even if later nodes could individually be
    resolved — because we cannot statically reason about what happens between
    the boundary and those nodes through an opaque segment.
    """
    resolved_prefix: List[Dict[str, Any]] = []
    unresolved_block: List[Dict[str, Any]] = []

    for row in rows:
        endpoint = row.get("endpoint", "") or ""
        is_unresolved = bool(row.get("is_unresolved")) or endpoint.startswith(
            UNRESOLVED_ENDPOINT_PREFIX
        )
        if unresolved_block or is_unresolved:
            unresolved_block.append(row)
        else:
            resolved_prefix.append(row)

    return resolved_prefix, unresolved_block


def summarize_unresolved_boundary(
    pm: Dict[str, Any],
    call_chain: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build metadata about the unresolved downstream black box, if any.

    The unresolved block is anchored at the last resolved endpoint (the
    "boundary"). Roles that the boundary allows are the ones that must be
    exercised at runtime against that boundary endpoint; a divergence from the
    boundary's expected result signals an inconsistency somewhere inside the
    opaque block. Roles that the boundary denies cannot be resolved statically.
    """
    roles = pm.get("roles", []) or []
    _, rows = collect_chain_permission_rows(pm, call_chain)
    resolved_prefix, unresolved_block = partition_chain_rows_by_resolution(rows)

    if not unresolved_block or not resolved_prefix or not roles:
        return {}

    boundary = resolved_prefix[-1]
    boundary_values = boundary.get("values", []) or []

    roles_requiring_runtime_tests: List[str] = []
    roles_unresolvable: List[str] = []
    roles_unknown_at_boundary: List[str] = []

    for r_idx, role in enumerate(roles):
        if r_idx >= len(boundary_values):
            continue
        v = boundary_values[r_idx]
        if v == 1:
            roles_requiring_runtime_tests.append(role)
        elif v == 0:
            roles_unresolvable.append(role)
        else:
            roles_unknown_at_boundary.append(role)

    return {
        "boundary_endpoint": boundary["endpoint"],
        "unresolved_endpoints": [row["endpoint"] for row in unresolved_block],
        "roles_requiring_runtime_tests": roles_requiring_runtime_tests,
        "roles_unresolvable": roles_unresolvable,
        "roles_unknown_at_boundary": roles_unknown_at_boundary,
        "guidance": (
            "Treat the unresolved downstream nodes as a single black box. "
            "Generate tests targeting the boundary endpoint with every role "
            "to observe responses; a divergence from the boundary's expected "
            "outcome indicates an inconsistency inside the black box. Roles "
            "denied at the boundary cannot be resolved statically."
        ),
    }


def dedupe_policy_inconsistencies(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()

    for item in items or []:
        if not isinstance(item, dict):
            continue
        details = item.get("details", {}) or {}
        key = (
            item.get("type"),
            item.get("role"),
            details.get("upstream_endpoint"),
            details.get("downstream_endpoint"),
            details.get("endpoint"),
            details.get("boundary_endpoint"),
            details.get("reason"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def extract_policy_inconsistency_roles(items: List[Dict[str, Any]]) -> List[str]:
    return sorted({item.get("role") for item in items or [] if item.get("role")})


def summarize_permissions(pm: Dict[str, Any], endpoint_id: str) -> Dict[str, Any]:
    """Summarizes permissions and detects potential inconsistencies for a specific endpoint.
    
    Analyzes the permission matrix for an endpoint to categorize roles as allowed,
    denied, or unknown. Also detects sensitive endpoints that may be incorrectly
    marked as public, which could indicate security policy violations.
    
    Args:
        pm (Dict[str, Any]): The permission matrix dictionary containing roles,
            endpoints, and permission values.
        endpoint_id (str): The ID of the endpoint to analyze.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - allowed_roles (List[str]): Roles with permission value 1
            - denied_roles (List[str]): Roles with permission value 0  
            - unknown_roles (List[str]): Roles with permission value -1
            - expected_status_by_role (Dict[str, str]): Mapping of roles to expected
              HTTP status codes (2xx, 403, UNKNOWN)
            - is_public_endpoint (bool): True if all roles have -1 value
            - policy_inconsistencies (List[Dict]): List of detected policy violations
              such as sensitive endpoints marked as public
    """
    row, roles = _row_for_endpoint(pm, endpoint_id)
    allowed, denied, unknown = [], [], []
    expected_status_by_role = {}
    policy_inconsistencies = []
    is_public_endpoint = False

    if row:
        # For an entry-point endpoint summary, -1 means "no permission check = public".
        row_summary = split_permission_row(roles, row, is_public_hint=True)
        allowed = row_summary["allowed_roles"]
        denied = row_summary["denied_roles"]
        unknown = row_summary["unknown_roles"]
        expected_status_by_role = row_summary["expected_status_by_role"]
        is_public_endpoint = row_summary["is_public_endpoint"]

        # Sensitivity is inferred from the endpoint URI in `infer_data_and_sensitivity`
        # (single source of truth). Doing it here against the endpoint_id produced
        # false positives because endpoint IDs include service names (e.g.
        # "configservice:<hash>" wrongly matched the INTERNAL keyword "config",
        # "userservice:<hash>" wrongly matched the PII keyword "user"), generating
        # stale PolicyExposure inconsistencies for endpoints that are actually
        # PUBLIC by URI.

    return {
        "allowed_roles": allowed,
        "denied_roles": denied,
        "unknown_roles": unknown,
        "expected_status_by_role": expected_status_by_role,
        "is_public_endpoint": is_public_endpoint,
        "policy_inconsistencies": policy_inconsistencies,
    }


def compute_chain_permissions(pm: Dict[str, Any], call_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes permissions for an entire call chain using the permission matrix.

    For each endpoint in the call chain that is present in the permission matrix,
    this function computes allowed/denied/unknown roles and then derives the set
    of roles that are allowed on *every* endpoint in the chain (i.e., roles that
    can successfully traverse the whole chain without an authorization failure).

    Returns a dictionary containing:
        - chain_allowed_roles: roles allowed at all covered endpoints in the chain
        - per_endpoint: mapping endpoint_id -> {allowed_roles, denied_roles, unknown_roles}
    """

    roles: List[str] = pm.get("roles", [])
    if not roles or not call_chain:
        return {
            "chain_allowed_roles": [],
            "chain_denied_roles": [],
            "chain_unknown_roles": [],
            "chain_expected_status_by_role": {},
            "per_endpoint": {},
        }

    per_endpoint: Dict[str, Dict[str, List[str]]] = {}
    roles, rows = collect_chain_permission_rows(pm, call_chain)

    for row_info in rows:
        # Resolved endpoints: -1 means "no restriction" = public access (is_public_hint=True).
        # Unresolved endpoints: -1 means "we have no data" = unknown (is_public_hint=False).
        is_public_hint = not row_info.get("is_unresolved", False)
        row_summary = split_permission_row(row_info["roles"], row_info["values"], is_public_hint=is_public_hint)
        per_endpoint[row_info["endpoint"]] = {
            "allowed_roles": row_summary["allowed_roles"],
            "denied_roles": row_summary["denied_roles"],
            "unknown_roles": row_summary["unknown_roles"],
        }

    if not per_endpoint:
        return {
            "chain_allowed_roles": [],
            "chain_denied_roles": [],
            "chain_unknown_roles": [],
            "chain_expected_status_by_role": {},
            "per_endpoint": {},
        }

    chain_allowed_roles: List[str] = []
    chain_denied_roles: List[str] = []
    chain_unknown_roles: List[str] = []
    chain_expected_status_by_role: Dict[str, str] = {}

    for role in roles:
        denied_somewhere = any(role in info["denied_roles"] for info in per_endpoint.values())
        allowed_somewhere = any(role in info["allowed_roles"] for info in per_endpoint.values())

        if denied_somewhere:
            chain_denied_roles.append(role)
            chain_expected_status_by_role[role] = "403"
        elif allowed_somewhere:
            chain_allowed_roles.append(role)
            chain_expected_status_by_role[role] = "2xx"
        else:
            chain_unknown_roles.append(role)
            chain_expected_status_by_role[role] = "UNKNOWN"

    return {
        "chain_allowed_roles": chain_allowed_roles,
        "chain_denied_roles": chain_denied_roles,
        "chain_unknown_roles": chain_unknown_roles,
        "chain_expected_status_by_role": chain_expected_status_by_role,
        "per_endpoint": per_endpoint,
    }


# =====================================================
#  POLICY INCONSISTENCY DETECTION (ENHANCED)
# =====================================================
def detect_policy_inconsistency(pm: Dict[str, Any], call_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detects and classifies authorization policy inconsistencies along a call chain.
    
    Analyzes consecutive endpoints in a call chain to identify authorization
    policy mismatches that could lead to security vulnerabilities such as
    privilege escalation or access control bypasses.
    
    Args:
        pm (Dict[str, Any]): The permission matrix containing roles, endpoints,
            and permission values.
        call_chain (List[Dict[str, Any]]): A list of dictionaries representing
            the call chain with endpoint information.
            
    Returns:
        List[Dict[str, Any]]: A list of inconsistency dictionaries, each containing:
            - role (str): The role affected by the inconsistency
            - type (str): The type of inconsistency (UnderPermissiveDownstream,
              OverPermissiveDownstream, UndefinedPolicy, RoleMismatch)
            - details (Dict): Information about upstream/downstream endpoints
              and their permission values
              
    Returns an empty list if no endpoints are found in the permission matrix
    or if the matrix data is incomplete.
    """

    roles = pm.get("roles", [])
    _, rows = collect_chain_permission_rows(pm, call_chain)

    if not roles or len(rows) < 2:
        return []

    resolved_prefix, unresolved_block = partition_chain_rows_by_resolution(rows)

    inconsistencies: List[Dict[str, Any]] = []

    # 1) Compare the entrypoint (root) only against *resolved* downstream
    #    endpoints. Unresolved nodes are skipped here because we cannot reason
    #    about them individually — they are handled as a single black box below.
    if len(resolved_prefix) >= 2:
        root = resolved_prefix[0]
        for r_idx, role in enumerate(roles):
            v_up = root["values"][r_idx]
            for down in resolved_prefix[1:]:
                v_down = down["values"][r_idx]

                if v_up == v_down:
                    continue

                if -1 in (v_up, v_down):
                    inconsistency_type = "UndefinedPolicy"
                elif v_up == 1 and v_down == 0:
                    inconsistency_type = "UnderPermissiveDownstream"
                elif v_up == 0 and v_down == 1:
                    inconsistency_type = "OverPermissiveDownstream"
                else:
                    inconsistency_type = "RoleMismatch"

                inconsistencies.append({
                    "role": role,
                    "type": inconsistency_type,
                    "details": {
                        "upstream_endpoint": root["endpoint"],
                        "downstream_endpoint": down["endpoint"],
                        "upstream_value": v_up,
                        "downstream_value": v_down,
                    },
                })

    # 2) Collapse the unresolved block into a single black box anchored at the
    #    last resolved endpoint (the boundary). Only roles that the boundary
    #    denies are raised statically as UnresolvedDownstream, because for
    #    those we cannot tell what happens inside the black box and cannot
    #    exercise them at runtime either. Roles the boundary allows are meant
    #    to be tested at runtime against the boundary endpoint; that guidance
    #    is captured via summarize_unresolved_boundary(), not here.
    if unresolved_block and resolved_prefix:
        boundary = resolved_prefix[-1]
        boundary_values = boundary.get("values", []) or []
        unresolved_endpoints = [row["endpoint"] for row in unresolved_block]

        for r_idx, role in enumerate(roles):
            if r_idx >= len(boundary_values):
                continue
            if boundary_values[r_idx] != 0:
                continue

            inconsistencies.append({
                "role": role,
                "type": "UnresolvedDownstream",
                "details": {
                    "boundary_endpoint": boundary["endpoint"],
                    "unresolved_endpoints": unresolved_endpoints,
                    "boundary_value": boundary_values[r_idx],
                    "reason": (
                        "Role denied at boundary endpoint; downstream "
                        "behavior inside the unresolved block cannot be "
                        "statically resolved."
                    ),
                },
            })

    return dedupe_policy_inconsistencies(inconsistencies)


# =====================================================
#  SCENARIO CATEGORIZATION & GENERATION
# =====================================================
def categorize_scenario(scenario_type: str,
                        inconsistencies: List[Dict[str, Any]],
                        sensitivity: str,
                        is_public: bool) -> str:
    """Derives a high-level scenario category for theoretical classification.

    This makes explicit the cross-product of:
    - scenario_type: EntryPointAuthorization vs DownstreamPolicyConsistency
    - presence and type of policy inconsistencies
    - sensitivity: FINANCIAL, PII, INTERNAL, PUBLIC
    - exposure: public vs restricted endpoint

    The result is a stable label that can be used in analysis, reporting and
    prompt template selection.
    """

    base = "ENTRYPOINT" if scenario_type == "EntryPointAuthorization" else "DOWNSTREAM"

    if not inconsistencies:
        inconsistency_part = "CONSISTENT"
    else:
        types = {i.get("type") for i in inconsistencies}
        if "PolicyExposure" in types and is_public:
            inconsistency_part = "POLICY_EXPOSURE"
        elif "OverPermissiveDownstream" in types:
            inconsistency_part = "OVER_PERMISSIVE"
        elif "UnderPermissiveDownstream" in types:
            inconsistency_part = "UNDER_PERMISSIVE"
        elif "UndefinedPolicy" in types:
            inconsistency_part = "UNDEFINED_POLICY"
        else:
            inconsistency_part = "ROLE_MISMATCH"

    sensitivity_part = sensitivity or "UNKNOWN"

    exposure_part = "PUBLIC" if is_public else "RESTRICTED"

    return f"{base}_{inconsistency_part}_{sensitivity_part}_{exposure_part}"


def generate_scenarios(av: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates structured test scenarios from all_vectors.json data.
    
    Processes each endpoint vector to create comprehensive test scenarios
    including permission analysis, policy inconsistency detection, and metadata
    for authorization testing.
    
    Args:
        av (Dict[str, Any]): The all_vectors data containing endpoint information,
            call chains, permission matrices, and statistics.
            
    Returns:
        List[Dict[str, Any]]: A list of scenario dictionaries, each containing:
            - scenario_id (str): Unique identifier for the scenario
            - type (str): EntryPointAuthorization or DownstreamPolicyConsistency
            - service_name (str): Name of the service
            - endpoint (str): The endpoint URI
            - method (str): HTTP method
            - allowed_roles/denied_roles/unknown_roles (List[str]): Permission categories
            - expected_status_by_role (Dict[str, str]): Role-to-status mapping
            - max_depth (int): Maximum call chain depth
            - total_calls (int): Total number of calls in the chain
            - policy_inconsistencies (List[Dict]): Detected authorization issues
            - policy_inconsistency_roles (List[str]): Roles with inconsistencies
            - expected_outcome (str): Description of expected test results
    """
    scenarios = []
    for endpoint_id, v in av["vectors"].items():
        ep = v["endpoint_info"]
        call_chain = v.get("call_chain", [])
        pm = v.get("permission_matrix", {})
        stats = v.get("statistics", {})

        scenario_type = classify_scenario(call_chain)
        perms_summary = summarize_permissions(pm, ep["id"])
        chain_perms = compute_chain_permissions(pm, call_chain)
        unresolved_boundary = summarize_unresolved_boundary(pm, call_chain)
        inconsistencies = dedupe_policy_inconsistencies(
            perms_summary["policy_inconsistencies"] + detect_policy_inconsistency(pm, call_chain)
        )

        if scenario_type == "DownstreamPolicyConsistency":
            allowed_roles = chain_perms.get("chain_allowed_roles") or perms_summary["allowed_roles"]
            denied_roles = chain_perms.get("chain_denied_roles") or perms_summary["denied_roles"]
            unknown_roles = chain_perms.get("chain_unknown_roles") or perms_summary["unknown_roles"]
            expected_status_by_role = (
                chain_perms.get("chain_expected_status_by_role")
                or perms_summary["expected_status_by_role"]
            )
        else:
            allowed_roles = perms_summary["allowed_roles"]
            denied_roles = perms_summary["denied_roles"]
            unknown_roles = perms_summary["unknown_roles"]
            expected_status_by_role = perms_summary["expected_status_by_role"]

        scenario = {
            "scenario_id": f"scn_{endpoint_id}",
            "type": scenario_type,
            "service_name": ep["service_name"],
            "endpoint": ep["uri"],
            "method": ep["method"],
            "allowed_roles": allowed_roles,
            "denied_roles": denied_roles,
            "unknown_roles": unknown_roles,
            "expected_status_by_role": expected_status_by_role,
            "chain_permissions": chain_perms.get("per_endpoint", {}),
            "max_depth": stats.get("max_depth_reached", 0),
            "total_calls": stats.get("total_calls", 0),
            "policy_inconsistencies": inconsistencies,
            "policy_inconsistency_roles": extract_policy_inconsistency_roles(inconsistencies),
            "unresolved_boundary": unresolved_boundary,
            "expected_outcome": (
                "Detect and explain inconsistency in downstream authorization policies"
                if inconsistencies and scenario_type == "DownstreamPolicyConsistency"
                else "Detect and explain entry-point authorization or exposure issue"
                if inconsistencies
                else "Return 2xx for allowed roles and 403 for denied roles"
            ),
        }

        # scenario_category will be enriched later, once sensitivity/authorization
        # exposure are known. For now we keep a placeholder for clarity.
        scenario["scenario_category"] = "UNSET"

        scenarios.append(scenario)
    return scenarios


# =====================================================
#  FEATURE EXTRACTION FROM ENDPOINTS.JSON
# =====================================================
def enrich_from_endpoints(scenario: Dict[str, Any], endpoints: list):
    """Enriches a scenario with detailed endpoint information from endpoints.json.
    
    Matches the scenario's endpoint URI against the endpoints data using multiple
    strategies (exact match, partial match, service-based fallback) and extracts
    authentication, authorization, API, and data schema information.
    
    Args:
        scenario (Dict[str, Any]): The scenario dictionary to enrich.
        endpoints (list): A list of endpoint data dictionaries from endpoints.json.
        
    Returns:
        Dict[str, Any]: The enriched scenario dictionary with additional fields:
            - auth (Dict): Authentication requirements and mechanisms
            - authorization (Dict): Enhanced authorization source and patterns
            - api (Dict): API characteristics like resource type and idempotency
            - data (Dict): Request/response schema information
            - notes (str): Warning message if endpoint is not found
    """
    ep_uri = scenario.get("endpoint")
    method = scenario.get("method")
    service_name = scenario.get("service_name", "").lower()

    # 1️⃣ Try exact match
    endpoint_data = next((e for e in endpoints if e.get("fullUri") == ep_uri and e.get("httpMethod") == method), None)

    # 2️⃣ Try partial match (URI contained or suffix match)
    if not endpoint_data and ep_uri:
        endpoint_data = next(
            (e for e in endpoints if ep_uri.endswith(e.get("fullUri", "")) or e.get("fullUri", "").endswith(ep_uri)),
            None,
        )

    # 3️⃣ Try service-based fallback (same service_name)
    if not endpoint_data:
        endpoint_data = next(
            (e for e in endpoints if service_name in e.get("fullUri", "").lower()),
            None,
        )

    # 4️⃣ If still nothing, skip safely
    if not endpoint_data:
        scenario["notes"] = f"[WARN] Endpoint not found for URI: {ep_uri}"
        return scenario

    # 5️⃣ Extract safely
    auth_info = endpoint_data.get("authentication") or {}
    mech_info = auth_info.get("mechanism") or {}
    authz_info = endpoint_data.get("authorization") or {}
    resp_info = endpoint_data.get("responseSchema") or {}

    scenario["auth"] = {
        "required": auth_info.get("required", False),
        "mechanism": mech_info.get("type", "Unknown"),
        "claims": mech_info.get("claims", []),
        "validation_rules": mech_info.get("validation", []),
    }

    scenario["authorization"].update({
        "source": authz_info.get("source"),
        "priority": authz_info.get("matchPriority"),
        "scope_pattern": authz_info.get("matchedPattern"),
        "public": authz_info.get("public", False),
    })

    scenario["api"] = {
        "resource_type": ep_uri.split("/")[-1] if ep_uri else None,
        "is_idempotent": endpoint_data.get("httpMethod") in ["GET", "HEAD", "OPTIONS"],
        "http_method": endpoint_data.get("httpMethod"),
    }

    status_map = (resp_info.get("statusCodes") or {})

    scenario["data"] = {
        "request_body_type": (endpoint_data.get("requestBodyDetail") or {}).get("type"),
        "response_type_name": (status_map.get("200") or {}).get("typeName"),
        "status_code_map": {k: (v or {}).get("typeName", "") for k, v in status_map.items()},
        "error_descriptions": {k: (v or {}).get("description", "") for k, v in status_map.items()},
    }

    # Expose raw parameter and URI metadata from endpoints.json for downstream consumers.
    scenario["pathParameterDetails"] = endpoint_data.get("pathParameterDetails", [])
    scenario["queryParameterDetails"] = endpoint_data.get("queryParameterDetails", [])
    scenario["fullUri"] = endpoint_data.get("fullUri")
    scenario["simplifiedUri"] = endpoint_data.get("simplifiedUri")
    scenario["curlExample"] = endpoint_data.get("curlExample")

    return scenario


# =====================================================
#  FEATURE EXTRACTION FROM COMPONENTS.JSON
# =====================================================
def enrich_from_components(scenario: Dict[str, Any], components: list):
    """Enriches a scenario with architectural information from components.json.
    
    Matches the scenario's service name against component data to extract
    architectural context including layer classification, dependencies, and
    entry point status.
    
    Args:
        scenario (Dict[str, Any]): The scenario dictionary to enrich.
        components (list): A list of component data dictionaries from components.json.
        
    Returns:
        Dict[str, Any]: The enriched scenario dictionary with additional
            'architecture' field containing:
            - layer (str): Component type or architectural layer
            - dependencies (List[str]): List of component dependencies
            - is_entrypoint (bool): True if component has no callers
            
        Returns the original scenario unchanged if no matching component is found.
    """
    service = scenario["service_name"]
    comp_data = next(
        (
            c for c in components
            if service in c.get("name", "") or service in c.get("id", "") or service in str(c)
        ),
        None,
    )
    if not comp_data:
        return scenario

    scenario["architecture"] = {
        "layer": comp_data.get("type", "Unknown"),
        "dependencies": comp_data.get("dependencies", []),
        "is_entrypoint": not bool(comp_data.get("calledBy", [])),
    }

    return scenario


def enrich_entity_schema_from_components(scenario: Dict[str, Any], components: list):
    """Attach an entity_schema for the request body type using components.json.

    Looks up the scenario's request_body_type (from endpoints.json enrichment)
    and collects all matching Field components (and their field-level
    annotations) for the corresponding entity class. The result is stored under
    scenario["entity_schema"], which can be used to construct valid request
    bodies in LLM prompts.

    If no request_body_type is present or no matching fields are found, the
    scenario is returned unchanged.
    """

    data_info = scenario.get("data") or {}
    entity_type = data_info.get("request_body_type")
    if not entity_type:
        return scenario

    service = scenario.get("service_name", "")

    fields: List[Dict[str, Any]] = []

    # Pre-collect annotations by class for efficient lookup
    annotations_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for c in components:
        if c.get("type") == "Annotation":
            cls = c.get("className")
            if cls:
                annotations_by_class.setdefault(cls, []).append(c)

    class_annotations = annotations_by_class.get(entity_type, [])

    for c in components:
        if c.get("type") != "Field":
            continue
        if c.get("className") != entity_type:
            continue
        micro = c.get("microservice", "")
        if service and service not in micro:
            continue

        field_name = c.get("name")
        field_type = c.get("effectiveTypeName") or c.get("fieldType")

        field_annotations: List[Dict[str, Any]] = []
        seen_ann: set[tuple] = set()
        for a in class_annotations:
            full_id = a.get("fullID") or a.get("fullId") or ""
            if f"#field:instance#field:{field_name}:" not in full_id:
                continue

            aname = a.get("name")
            attrs = a.get("attributes", {}) or {}

            # Build a hashable key (annotation name + sorted attributes) to
            # avoid duplicates when components.json contains repeated entries
            # for the same field-level annotation.
            key = (aname, tuple(sorted(attrs.items())))
            if key in seen_ann:
                continue
            seen_ann.add(key)

            field_annotations.append({
                "name": aname,
                "attributes": attrs,
            })

        fields.append({
            "name": field_name,
            "type": field_type,
            "annotations": field_annotations,
        })

    if not fields:
        return scenario

    scenario["entity_schema"] = {
        "entity_name": entity_type,
        "service_name": service,
        "fields": fields,
    }

    return scenario

# infer sensitivity and data awareness from endpoint
def infer_data_and_sensitivity(scenario: Dict[str, Any]):
    """Infers data awareness level and endpoint sensitivity from scenario characteristics.
    
    Analyzes the endpoint URI and call chain depth to determine the sensitivity
    type (FINANCIAL, PII, INTERNAL, PUBLIC) and data awareness level (HIGH/LOW).
    Also detects policy violations where sensitive endpoints are marked as public.
    
    Args:
        scenario (Dict[str, Any]): The scenario dictionary to analyze and enrich.
            Must contain 'endpoint' and 'max_depth' fields, and 'authorization'
            field with 'public' key.
            
    Returns:
        Dict[str, Any]: The enriched scenario dictionary with additional fields:
            - data_awareness_level (str): "HIGH" for multi-call scenarios, "LOW" otherwise
            - llm_prompt_type (str): "CHAIN_OF_THOUGHT" for HIGH, "SIMPLE" for LOW
            - sensitivity_type (str): One of "FINANCIAL", "PII", "INTERNAL", "PUBLIC"
            - policy_inconsistencies (List[Dict]): Additional violations if sensitive
              endpoints are marked as public
    """
    uri = scenario.get("endpoint", "").lower()
    depth = scenario.get("max_depth", 0)

    # Data awareness
    if depth > 0:
        scenario["data_awareness_level"] = "HIGH"
        scenario["llm_prompt_type"] = "CHAIN_OF_THOUGHT"
    else:
        scenario["data_awareness_level"] = "LOW"
        scenario["llm_prompt_type"] = "SIMPLE"

    # Sensitivity inference
    if any(k in uri for k in ["pay", "refund", "payment"]):
        scenario["sensitivity_type"] = "FINANCIAL"
    elif any(k in uri for k in ["user", "profile", "personal", "account"]):
        scenario["sensitivity_type"] = "PII"
    elif any(k in uri for k in ["admin", "config", "report", "manage"]):
        scenario["sensitivity_type"] = "INTERNAL"
    else:
        scenario["sensitivity_type"] = "PUBLIC"

    if scenario["sensitivity_type"] in ["FINANCIAL", "PII", "INTERNAL"] and scenario["authorization"].get("public", False):
        scenario.setdefault("policy_inconsistencies", []).append({
            "role": "PUBLIC",
            "type": "PolicyExposure",
            "details": {"reason": "Sensitive endpoint marked as PUBLIC (-1)"}
        })
        scenario["policy_inconsistencies"] = dedupe_policy_inconsistencies(scenario["policy_inconsistencies"])

    return scenario


# =====================================================
#  BUILD LLM PROMPT CONTEXT
# =====================================================
def select_prompt_template_id(scenario: Dict[str, Any]) -> str:
    """Selects a prompt template identifier based on scenario classification.

    Uses scenario_category together with scenario type and data awareness level
    to map each scenario into a stable prompt template class that the Test
    Generator/LLM can target.
    """

    category = scenario.get("scenario_category", "UNSET") or "UNSET"
    scenario_type = scenario.get("type")
    data_level = scenario.get("data_awareness_level", "LOW")

    if category == "UNSET":
        return "generic_authorization"

    # Entry-point focused templates
    if scenario_type == "EntryPointAuthorization":
        if "POLICY_EXPOSURE" in category:
            return "entrypoint_public_sensitive"
        return "entrypoint_status_matrix"

    # Downstream consistency templates
    if "CONSISTENT" in category:
        return "downstream_consistent_path"

    if data_level == "HIGH":
        return "downstream_inconsistency_cot"
    return "downstream_inconsistency_simple"


def build_prompt_context(scenario: Dict[str, Any]):
    """Builds LLM prompt context with goals and sensitivity information.
    
    Creates structured prompt context for LLM-based test generation based on
    scenario type, data awareness level, and sensitivity characteristics.
    
    Args:
        scenario (Dict[str, Any]): The scenario dictionary to build context for.
            Should contain 'type', 'data_awareness_level', and 'sensitivity_type' fields.
            
    Returns:
        Dict[str, Any]: The scenario dictionary with added 'prompt_context' field:
            - goals (List[str]): List of testing goals tailored to scenario type
            - prompt_type (str): LLM prompt type (CHAIN_OF_THOUGHT or SIMPLE)
            - sensitivity (str): Endpoint sensitivity classification
    """
    is_downstream = scenario.get("type") == "DownstreamPolicyConsistency"
    data_level = scenario.get("data_awareness_level", "LOW")

    base_goals = [
        "Validate role-based access control enforcement.",
        "Check that PUBLIC endpoints do not expose sensitive resources (PII/Financial)."
    ]
    if is_downstream:
        base_goals.append("Detect inconsistent authorization policies across services.")
        if data_level == "HIGH":
            base_goals.append("Ensure test data enables full downstream traversal without mid-chain exceptions.")

    template_id = scenario.get("prompt_template_id", "generic_authorization")

    # Build a focused view of parameters that the LLM prompt should receive.
    # This makes explicit which scenario fields are considered inputs per
    # template class.
    if template_id == "entrypoint_status_matrix":
        parameters = {
            "scenario_id": scenario.get("scenario_id"),
            "endpoint": scenario.get("endpoint"),
            "method": scenario.get("method"),
            "service_name": scenario.get("service_name"),
            "allowed_roles": scenario.get("allowed_roles", []),
            "denied_roles": scenario.get("denied_roles", []),
            "expected_status_by_role": scenario.get("expected_status_by_role", {}),
        }
    elif template_id == "entrypoint_public_sensitive":
        parameters = {
            "scenario_id": scenario.get("scenario_id"),
            "endpoint": scenario.get("endpoint"),
            "method": scenario.get("method"),
            "service_name": scenario.get("service_name"),
            "sensitivity_type": scenario.get("sensitivity_type"),
            "authorization": scenario.get("authorization", {}),
            "policy_inconsistencies": scenario.get("policy_inconsistencies", []),
        }
    elif template_id == "downstream_consistent_path":
        parameters = {
            "scenario_id": scenario.get("scenario_id"),
            "service_name": scenario.get("service_name"),
            "endpoint": scenario.get("endpoint"),
            "method": scenario.get("method"),
            "max_depth": scenario.get("max_depth", 0),
            "total_calls": scenario.get("total_calls", 0),
            "allowed_roles": scenario.get("allowed_roles", []),
        }
    elif template_id in ("downstream_inconsistency_cot", "downstream_inconsistency_simple"):
        parameters = {
            "scenario_id": scenario.get("scenario_id"),
            "service_name": scenario.get("service_name"),
            "endpoint": scenario.get("endpoint"),
            "method": scenario.get("method"),
            "max_depth": scenario.get("max_depth", 0),
            "total_calls": scenario.get("total_calls", 0),
            "policy_inconsistencies": scenario.get("policy_inconsistencies", []),
            "policy_inconsistency_roles": scenario.get("policy_inconsistency_roles", []),
        }
    else:
        # Generic fallback with core identification fields.
        parameters = {
            "scenario_id": scenario.get("scenario_id"),
            "service_name": scenario.get("service_name"),
            "endpoint": scenario.get("endpoint"),
            "method": scenario.get("method"),
            "type": scenario.get("type"),
        }

    scenario["prompt_context"] = {
        "goals": base_goals,
        "prompt_type": scenario.get("llm_prompt_type", "SIMPLE"),
        "sensitivity": scenario.get("sensitivity_type", "UNKNOWN"),
        "scenario_category": scenario.get("scenario_category", "UNSET"),
        "template_id": template_id,
        "parameters": parameters,
    }

    return scenario


def build_generation_audit_report(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    template_counter: Counter[str] = Counter()
    depth_counter: Counter[int] = Counter()
    total_calls_counter: Counter[int] = Counter()
    policy_type_counter: Counter[str] = Counter()
    missing_role_coverage: List[Dict[str, Any]] = []

    for scenario in scenarios:
        type_counter[scenario.get("type", "UNKNOWN")] += 1
        category_counter[scenario.get("scenario_category", "UNSET")] += 1
        template_counter[scenario.get("prompt_template_id", "UNKNOWN")] += 1
        depth_counter[scenario.get("max_depth", 0)] += 1
        total_calls_counter[scenario.get("total_calls", 0)] += 1

        for item in scenario.get("policy_inconsistencies", []) or []:
            policy_type_counter[item.get("type", "UNKNOWN")] += 1

        expected_roles = set((scenario.get("expected_status_by_role") or {}).keys())
        observed_roles = (
            set(scenario.get("allowed_roles", []) or [])
            | set(scenario.get("denied_roles", []) or [])
            | set(scenario.get("unknown_roles", []) or [])
        )
        if expected_roles != observed_roles:
            missing_role_coverage.append({
                "scenario_id": scenario.get("scenario_id"),
                "method": scenario.get("method"),
                "endpoint": scenario.get("endpoint"),
                "expected_roles": sorted(expected_roles),
                "observed_roles": sorted(observed_roles),
            })

    return {
        "total_scenarios": len(scenarios),
        "scenarios_per_type": dict(type_counter.most_common()),
        "scenarios_per_category": dict(category_counter.most_common()),
        "scenarios_per_prompt_template": dict(template_counter.most_common()),
        "scenarios_per_depth": dict(sorted(depth_counter.items())),
        "scenarios_per_total_calls": dict(sorted(total_calls_counter.items())),
        "policy_inconsistency_types": dict(policy_type_counter.most_common()),
        "missing_role_coverage_count": len(missing_role_coverage),
        "missing_role_coverage_examples": missing_role_coverage[:20],
    }


# =====================================================
#  MAIN PIPELINE
# =====================================================
def main():
    """Main pipeline function for generating LLM-ready test scenarios.
    
    Orchestrates the complete scenario generation process:
    1. Loads input data files (all_vectors.json, endpoints.json, components.json)
    2. Generates base scenarios with permission analysis
    3. Enriches scenarios with endpoint and component metadata
    4. Infers data awareness and sensitivity characteristics
    5. Builds LLM prompt context for test generation
    6. Extracts IR features for advanced analysis
    7. Outputs final scenarios to scenarios_llm_ready.json
    
    The function processes all vectors from the input data and creates a
    comprehensive test scenario dataset suitable for LLM-based authorization
    testing.
    
    Side effects:
        - Creates/overwrites 'output/llm-scenarios/scenarios_llm_ready.json' with generated scenarios
        - Prints completion message with scenario count to stdout
    """

    parser = argparse.ArgumentParser(description="Generate LLM-ready authorization scenarios.")
    parser.add_argument(
        "--template-id",
        dest="template_id",
        help="Filter scenarios by prompt_template_id before writing output.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional output file path for the generated scenarios.",
    )
    parser.add_argument(
        "--list-templates",
        dest="list_templates",
        action="store_true",
        help="List distinct prompt_template_id values present in the generated scenarios.",
    )
    parser.add_argument(
        "--audit-report",
        dest="audit_report_path",
        help="Optional JSON file path to write a generation audit summary.",
    )
    parser.add_argument(
        "--base-path",
        dest="base_path",
        default="yas/main",
        help="Dataset base folder containing all_vectors.json/endpoints.json/components.json.",
    )
    args = parser.parse_args()

    # BASE_PATH = args.base_path
    BASE_PATHS = [
        "train-ticket-aitest/first-inconsistency-injection",
        "train-ticket-aitest/master",
        "train-ticket-aitest/second-inconsistency-injection",
        "yas/main",
        "yas/first-injection",
        "yas/second-injection"
    ]

    for BASE_PATH in BASE_PATHS:
        logging.info(f"Generating scenarios for {BASE_PATH}")

        all_vectors = load_json(f"{BASE_PATH}/all_vectors.json")
        endpoints = load_json(f"{BASE_PATH}/endpoints.json")
        components = load_json(f"{BASE_PATH}/components.json")

        base_scenarios = generate_scenarios(all_vectors)
        enriched_scenarios = []

        for s in base_scenarios:
            s["authorization"] = {"required_roles": s.get("allowed_roles", [])}
            s = enrich_from_endpoints(s, endpoints)
            s = enrich_from_components(s, components)
            s = enrich_entity_schema_from_components(s, components)
            s = infer_data_and_sensitivity(s)
            s["policy_inconsistencies"] = dedupe_policy_inconsistencies(s.get("policy_inconsistencies", []))
            s["policy_inconsistency_roles"] = extract_policy_inconsistency_roles(s.get("policy_inconsistencies", []))

            # Now that sensitivity and exposure (public vs restricted) are known,
            # derive a high-level theoretical scenario category.
            s["scenario_category"] = categorize_scenario(
                s.get("type"),
                s.get("policy_inconsistencies", []),
                s.get("sensitivity_type"),
                s.get("authorization", {}).get("public", False),
            )

            # Select a concrete prompt template class for this scenario.
            s["prompt_template_id"] = select_prompt_template_id(s)

            s = build_prompt_context(s)
            
            ir_features = analyze_ir_node(s)
            s.update({
                "handles_pii": ir_features["handles_pii"],
                "pii_evidence": ir_features["pii_evidence"],
                "business_logic_constraints": ir_features["business_logic_constraints"]
            })
            enriched_scenarios.append(s)

        if args.list_templates:
            templates = sorted({s.get("prompt_template_id", "") for s in enriched_scenarios})
            print("Available prompt_template_id values:")
            for t in templates:
                if t:
                    print(f"- {t}")

        # Optional filtering by template_id for Test Generator integration.
        if args.template_id:
            enriched_scenarios = [
                s for s in enriched_scenarios
                if s.get("prompt_template_id") == args.template_id
            ]

        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = f"{BASE_PATH}/scenarios_llm_ready.json"

        audit_report = build_generation_audit_report(enriched_scenarios)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_scenarios, f, indent=2, ensure_ascii=False)
        if args.audit_report_path:
            with open(args.audit_report_path, "w", encoding="utf-8") as f:
                json.dump(audit_report, f, indent=2, ensure_ascii=False)
            print(f"📊 Generation audit report saved to {args.audit_report_path}")
        print(
            "📈 Generation summary: "
            f"types={audit_report['scenarios_per_type']} "
            f"missing_role_coverage={audit_report['missing_role_coverage_count']}"
        )
        print(f"✅ LLM-ready scenarios saved to {output_path} ({len(enriched_scenarios)} entries)")


if __name__ == "__main__":
    main()

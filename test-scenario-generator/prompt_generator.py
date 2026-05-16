"""Prompt Generator for Java test creation based on LLM-ready scenarios.

This module implements state-of-the-art prompt engineering techniques for LLM-based
test generation, including:
- Chain-of-Thought (CoT) prompting for complex scenarios
- Few-shot prompting with high-quality test examples
- Role-based prompting with specialized personas
- Structured prompting with clear sections
- Adaptive prompt executor using LangChain for long distributed paths
- Context window management for scenarios with extensive call chains
- Business intent and path structure analysis
"""

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# LangChain imports for advanced prompt chaining
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

LANGCHAIN_AVAILABLE = True


class MockLLMChain:
    """Mock LLMChain that returns the prompt template for testing without a real LLM."""
    
    def __init__(self, prompt: PromptTemplate, output_key: str):
        self.prompt = prompt
        self.output_key = output_key
    
    def run(self, **kwargs) -> Dict[str, str]:
        """Return the formatted prompt as the 'output'."""
        try:
            formatted = self.prompt.format(**kwargs)
            return {self.output_key: formatted}
        except Exception:
            # If formatting fails, return the template
            return {self.output_key: self.prompt.template}


class PromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    @abstractmethod
    def build(self, scenario: Dict[str, Any]) -> str:
        """Build a prompt for the given scenario."""
        pass


class FewShotExampleLibrary:
    """Library of high-quality test examples for few-shot prompting."""
    
    @classmethod
    def get_examples_for_scenario(cls, scenario: Dict[str, Any]) -> str:
        """Retrieve relevant examples based on scenario type."""
        scenario_type = scenario.get("type", "")
        category = scenario.get("scenario_category", "")
        
        if "POLICY_EXPOSURE" in category:
            return cls._get_policy_exposure_example()
        elif scenario_type == "DownstreamPolicyConsistency":
            return cls._get_downstream_example()
        else:
            return cls._get_entrypoint_example()
    
    @classmethod
    def _get_entrypoint_example(cls) -> str:
        return """
Example Test Structure:
```java
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class OrderControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Value("${test.jwt.admin}")
    private String adminToken;
    
    @Value("${test.jwt.user}")
    private String userToken;
    
    @Nested
    @DisplayName("Authorization - Role-based Access Control")
    class AuthorizationTests {
        
        @Test
        @DisplayName("GET /orders as ADMIN returns 200 with all orders")
        void getOrders_asAdmin_returns200() throws Exception {
            mockMvc.perform(get("/api/v1/orders")
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + adminToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.orders").exists());
        }
        
        @Test
        @DisplayName("GET /orders as USER returns 200 with own orders only")
        void getOrders_asUser_returns200() throws Exception {
            mockMvc.perform(get("/api/v1/orders")
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + userToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.orders").exists());
        }
        
        @Test
        @DisplayName("GET /orders without auth returns 401")
        void getOrders_withoutAuth_returns401() throws Exception {
            mockMvc.perform(get("/api/v1/orders"))
                .andExpect(status().isUnauthorized());
        }
    }
}
```
"""
    
    @classmethod
    def _get_downstream_example(cls) -> str:
        return """
Example Test Structure for Downstream Scenarios:
```java
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class BookingFlowTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Value("${test.jwt.user}")
    private String userToken;
    
    /**
     * Tests the complete booking flow across multiple services:
     * BookingService -> PaymentService -> SeatAllocationService
     */
    @Nested
    @DisplayName("POST /bookings - Distributed Authorization Consistency")
    class BookingFlowAuthorizationTests {
        
        @Test
        @DisplayName("Complete booking flow with USER role - all services authorize consistently")
        void completeBookingFlow_userRole_allServicesAuthorize() throws Exception {
            String validBookingRequest = buildValidBookingRequest();
            
            mockMvc.perform(post("/api/v1/bookings")
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + userToken)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(validBookingRequest))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.bookingId").exists())
                .andExpect(jsonPath("$.paymentStatus").value("PROCESSED"))
                .andExpect(jsonPath("$.seatStatus").value("CONFIRMED"));
        }
    }
}
```
"""
    
    @classmethod
    def _get_policy_exposure_example(cls) -> str:
        return """
Example Test Structure for Security Vulnerabilities:
```java
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class PaymentEndpointSecurityTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    /**
     * SECURITY VULNERABILITY: Payment endpoint configured as PUBLIC.
     * Endpoint: POST /api/v1/payments/process
     * Risk: Unauthenticated payment processing
     */
    @Nested
    @DisplayName("POST /payments/process - PUBLIC ENDPOINT SECURITY ISSUE")
    class PaymentEndpointVulnerabilityTests {
        
        @Test
        @DisplayName("CRITICAL: Payment endpoint accessible without authentication")
        void paymentEndpoint_publicAccess_documentVulnerability() throws Exception {
            String paymentRequest = buildPaymentRequest();
            
            // This demonstrates the security issue: no auth header required
            mockMvc.perform(post("/api/v1/payments/process")
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(paymentRequest))
                .andExpect(status().isOk())  // Vulnerability: returns 200 without auth
                .andDo(result -> {
                    System.err.println("SECURITY ISSUE: Payment endpoint accepts" +
                        " requests without authentication.");
                });
        }
    }
}
```
"""


class BusinessIntentAnalyzer:
    """Analyzes and extracts business intent from API endpoints."""
    
    DOMAIN_PATTERNS = {
        "payment": ["pay", "payment", "refund", "transaction", "billing", "invoice"],
        "user_management": ["user", "profile", "account", "customer", "identity"],
        "booking": ["book", "reservation", "ticket", "seat", "schedule"],
        "inventory": ["product", "inventory", "stock", "warehouse", "catalog"],
        "administration": ["admin", "config", "setting", "management", "report"],
        "logistics": ["delivery", "shipping", "tracking", "route", "transport"],
    }
    
    def analyze(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business context from scenario."""
        endpoint = scenario.get("endpoint", "").lower()
        method = scenario.get("method", "")
        service_name = scenario.get("service_name", "").lower()
        
        # Determine domain
        domain = self._detect_domain(endpoint, service_name)
        
        # Determine business operation
        operation = self._detect_operation(method)
        
        # Determine resource type
        resource = self._detect_resource_type(endpoint)
        
        # Determine sensitivity
        sensitivity = self._determine_sensitivity(domain, endpoint)
        
        return {
            "domain": domain,
            "business_operation": operation,
            "resource_type": resource,
            "sensitivity_level": sensitivity,
        }
    
    def _detect_domain(self, endpoint: str, service_name: str) -> str:
        """Detect business domain from endpoint and service name."""
        combined = f"{endpoint} {service_name}"
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            if any(pattern in combined for pattern in patterns):
                return domain
        
        return "general"
    
    def _detect_operation(self, method: str) -> str:
        """Detect business operation type."""
        method_ops = {
            "POST": "CREATE",
            "GET": "READ",
            "PUT": "UPDATE",
            "PATCH": "PARTIAL_UPDATE",
            "DELETE": "DELETE"
        }
        return method_ops.get(method.upper(), "CUSTOM")
    
    def _detect_resource_type(self, endpoint: str) -> str:
        """Extract primary resource type from endpoint."""
        parts = endpoint.rstrip("/").split("/")
        if parts:
            for part in reversed(parts):
                if part and not part.startswith("{"):
                    return part.replace("{", "").replace("}", "")
        return "resource"
    
    def _determine_sensitivity(self, domain: str, endpoint: str) -> str:
        """Determine data sensitivity level."""
        endpoint_lower = endpoint.lower()
        
        high_sensitivity = ["payment", "pay", "user", "profile", "admin", "config"]
        medium_sensitivity = ["booking", "order", "account"]
        
        if any(term in endpoint_lower for term in high_sensitivity):
            return "HIGH"
        if any(term in endpoint_lower for term in medium_sensitivity):
            return "MEDIUM"
        return "LOW"


class PathStructureAnalyzer:
    """Analyzes the structure of API paths."""
    
    def analyze(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze path structure from scenario."""
        endpoint = scenario.get("endpoint", "")
        path_params = scenario.get("pathParameterDetails", [])
        method = scenario.get("method", "")
        
        # Parse segments
        segments = [s for s in endpoint.split("/") if s]
        
        # Build resource hierarchy
        resource_hierarchy = []
        for seg in segments:
            if not seg.startswith("{"):
                resource_hierarchy.append(seg)
        
        # Determine action type
        action_type = self._determine_action_type(method, segments)
        
        # Calculate nesting depth
        nesting_depth = len([s for s in segments if s.startswith("{")])
        
        return {
            "segments": segments,
            "path_variables": path_params,
            "resource_hierarchy": resource_hierarchy,
            "action_type": action_type,
            "nesting_depth": nesting_depth,
        }
    
    def _determine_action_type(self, method: str, segments: List[str]) -> str:
        """Determine the CRUD action type."""
        last_segment = segments[-1] if segments else ""
        
        action_indicators = {
            "CREATE": ["create", "new", "add"],
            "UPDATE": ["update", "edit", "modify"],
            "DELETE": ["delete", "remove", "cancel"],
            "SEARCH": ["search", "find", "query"],
        }
        
        for action, indicators in action_indicators.items():
            if any(ind in last_segment.lower() for ind in indicators):
                return action
        
        return {
            "POST": "CREATE",
            "GET": "READ",
            "PUT": "UPDATE",
            "PATCH": "PARTIAL_UPDATE",
            "DELETE": "DELETE"
        }.get(method.upper(), "CUSTOM")


class EnhancedPromptBuilder(PromptBuilder):
    """Enhanced prompt builder with business context and structured output."""
    
    def __init__(self):
        self.business_analyzer = BusinessIntentAnalyzer()
        self.path_analyzer = PathStructureAnalyzer()
        self.example_library = FewShotExampleLibrary()
    
    def build(self, scenario: Dict[str, Any]) -> str:
        """Build an enhanced prompt for the given scenario."""
        # Analyze business and path context
        business_context = self.business_analyzer.analyze(scenario)
        path_structure = self.path_analyzer.analyze(scenario)
        
        # Build prompt components
        sections = [
            self._build_role_section(),
            self._build_task_section(scenario),
            self._build_business_context_section(business_context),
            self._build_path_structure_section(path_structure),
            self._build_scenario_section(scenario),
            self._build_examples_section(scenario),
            self._build_guidelines_section(scenario),
            self._build_output_specification(scenario),
        ]
        
        return "\n\n".join(filter(None, sections))
    
    def _build_role_section(self) -> str:
        """Build the role definition section using persona-based prompting."""
        return """# ROLE AND EXPERTISE

You are an **Expert Software Test Engineer** specializing in:
- Java enterprise application testing
- Spring Boot and Spring Security test automation
- REST API contract and authorization testing
- Microservices integration testing
- Security vulnerability detection and testing

Your expertise includes:
- Designing comprehensive test suites that validate both functional and non-functional requirements
- Implementing robust authorization tests using real JWT tokens (not mock users)
- Creating maintainable, well-documented test code following industry best practices
- Detecting and documenting security inconsistencies in distributed systems

You write tests that serve as **living documentation** and **security specifications** for the API."""
    
    def _build_task_section(self, scenario: Dict[str, Any]) -> str:
        """Build the task definition section."""
        template_id = scenario.get("prompt_template_id", "")
        
        task_descriptions = {
            "entrypoint_status_matrix": """## PRIMARY TASK

Generate comprehensive JUnit 5 authorization tests for a Spring Boot REST endpoint. 

**Focus**: Validate that the endpoint correctly enforces role-based access control (RBAC) as defined in the permission matrix. For each role, verify the expected HTTP status code (2xx for allowed, 403 for denied).""",
            
            "entrypoint_public_sensitive": """## PRIMARY TASK

Generate security-focused JUnit 5 tests that document and validate a **critical security vulnerability**: a sensitive endpoint (financial/PII/internal) that is configured as PUBLIC (accessible without authentication).

**Focus**: 
1. Demonstrate that the endpoint is accessible without authentication
2. Document the security risk with clear comments
3. Provide tests showing what the correct behavior should be after fixing the vulnerability
4. Serve as evidence for security audit and remediation planning""",
            
            "downstream_consistent_path": """## PRIMARY TASK

Generate integration-level JUnit 5 tests for a multi-service call chain that maintains **consistent authorization policies** across all participating services.

**Focus**:
1. Validate that requests with allowed roles can traverse the entire call chain
2. Ensure test data is realistic and enables full downstream execution
3. Verify that authorization decisions are coherent across service boundaries
4. Test both success paths and graceful failure handling""",
            
            "downstream_inconsistency_cot": """## PRIMARY TASK

Generate JUnit 5 tests using **Chain-of-Thought reasoning** to detect and document **authorization policy inconsistencies** in a distributed call chain.

**Focus**:
1. Identify where in the call chain authorization policies diverge
2. Document the inconsistency type (over-permissive, under-permissive, undefined)
3. Create tests that demonstrate both the baseline (expected) and divergent (actual) behavior
4. Provide clear explanations of the security implications
5. Use realistic data that triggers the inconsistency""",
            
            "downstream_inconsistency_simple": """## PRIMARY TASK

Generate JUnit 5 tests that highlight **authorization inconsistencies** in a downstream call chain.

**Focus**:
1. Create focused tests showing the status code differences between entry point and downstream
2. Document the inconsistency with clear comments
3. Prioritize clarity over comprehensive path simulation""",
        }
        
        return task_descriptions.get(
            template_id, 
            """## PRIMARY TASK

Generate JUnit 5 authorization tests for a Spring Boot REST endpoint based on the provided scenario metadata. Focus on validating role-based access control and expected HTTP status codes."""
        )
    
    def _build_business_context_section(self, context: Dict[str, Any]) -> str:
        """Build the business context section."""
        return f"""## BUSINESS CONTEXT

**Domain**: {context['domain'].upper()}
**Operation Type**: {context['business_operation']}
**Primary Resource**: {context['resource_type']}
**Data Sensitivity Level**: {context['sensitivity_level']}

**Testing Implications**:
- Authorization tests must respect the sensitivity level of the data
- Tests should validate compliance with stated requirements
- Security boundaries must be explicitly tested and documented"""
    
    def _build_path_structure_section(self, structure: Dict[str, Any]) -> str:
        """Build the path structure analysis section."""
        path_vars = structure.get("path_variables", [])
        path_vars_str = "\n".join(
            f"- `{pv.get('name', 'unknown')}`: {pv.get('type', 'String')}"
            for pv in path_vars
        ) if path_vars else "- None"
        
        hierarchy = structure.get("resource_hierarchy", [])
        hierarchy_str = " → ".join(hierarchy) if hierarchy else "root"
        
        return f"""## API PATH STRUCTURE

**Resource Hierarchy**: {hierarchy_str}
**Action Type**: {structure['action_type']}
**Path Nesting Depth**: {structure['nesting_depth']}

**Path Segments**: {' / '.join(structure['segments'])}

**Path Variables**:
{path_vars_str}

**Structural Analysis**:
This endpoint follows a {'nested' if structure['nesting_depth'] > 0 else 'flat'} resource structure.
The action represents a {structure['action_type']} operation on the {hierarchy[-1] if hierarchy else 'resource'} resource."""
    
    def _build_scenario_section(self, scenario: Dict[str, Any]) -> str:
        """Build the scenario-specific section."""
        ctx = scenario.get("prompt_context", {})
        params = ctx.get("parameters", {})
        
        template_id = ctx.get("template_id", "generic_authorization")
        
        scenario_id = scenario.get("scenario_id", params.get("scenario_id", "unknown"))
        endpoint = scenario.get("endpoint", params.get("endpoint", ""))
        method = scenario.get("method", params.get("method", ""))
        service = scenario.get("service_name", params.get("service_name", ""))
        
        base_info = f"""## SCENARIO SPECIFICATION

**Scenario ID**: {scenario_id}
**Template Class**: {template_id}
**Scenario Type**: {scenario.get("type", "unknown")}
**Scenario Category**: {scenario.get("scenario_category", "UNSET")}

**Target Endpoint**:
- Service: {service}
- HTTP Method: {method}
- Endpoint Path: {endpoint}
- Full URI: {scenario.get("fullUri", endpoint)}
"""
        
        # Add template-specific details
        if template_id == "entrypoint_status_matrix":
            allowed = params.get("allowed_roles", scenario.get("allowed_roles", []))
            denied = params.get("denied_roles", scenario.get("denied_roles", []))
            expected = params.get("expected_status_by_role", scenario.get("expected_status_by_role", {}))
            
            expected_str = "\n".join(f"  - {role}: {status}" for role, status in expected.items())
            
            base_info += f"""
**Authorization Matrix**:
- Allowed Roles: {', '.join(allowed) if allowed else 'None'}
- Denied Roles: {', '.join(denied) if denied else 'None'}

**Expected Status by Role**:
{expected_str}
"""
        
        elif template_id == "entrypoint_public_sensitive":
            sensitivity = params.get("sensitivity_type", scenario.get("sensitivity_type", "UNKNOWN"))
            authz = params.get("authorization", scenario.get("authorization", {}))
            
            base_info += f"""
**Security Vulnerability Details**:
- Sensitivity Type: {sensitivity}
- Public Endpoint: {authz.get('public', False)}
- Required Roles (if not public): {', '.join(authz.get('required_roles', []))}

**Security Implications**:
This endpoint handles {sensitivity} data but is accessible without authentication.
This represents a critical security vulnerability requiring immediate attention.
"""
        
        elif template_id in ("downstream_consistent_path", "downstream_inconsistency_cot", "downstream_inconsistency_simple"):
            max_depth = params.get("max_depth", scenario.get("max_depth", 0))
            total_calls = params.get("total_calls", scenario.get("total_calls", 0))
            chain_perms = scenario.get("chain_permissions", {})
            inconsistencies = params.get("policy_inconsistencies", scenario.get("policy_inconsistencies", []))
            
            base_info += f"""
**Call Chain Analysis**:
- Max Depth: {max_depth}
- Total Services in Chain: {total_calls}
- Chain Endpoints: {', '.join(chain_perms.keys())}
- Inconsistencies Detected: {len(inconsistencies)}
"""
        
        # Add entity schema if available
        entity_block = self._format_entity_schema(scenario)
        if entity_block:
            base_info += f"""
{entity_block}
"""
        
        # Add curl example if available
        curl = scenario.get("curlExample", "")
        if curl:
            base_info += f"""
**cURL Example**:
```bash
{curl}
```
"""
        
        return base_info
    
    def _build_examples_section(self, scenario: Dict[str, Any]) -> str:
        """Build the few-shot examples section."""
        template_id = scenario.get("prompt_template_id", "")
        
        if template_id not in (
            "entrypoint_status_matrix", 
            "entrypoint_public_sensitive",
            "downstream_consistent_path",
            "downstream_inconsistency_cot",
            "downstream_inconsistency_simple"
        ):
            return ""
        
        examples = self.example_library.get_examples_for_scenario(scenario)
        
        return f"""## REFERENCE EXAMPLES

The following examples demonstrate the expected test structure, naming conventions, and documentation style:

{examples}

**Key Patterns to Follow**:
1. Use `@Nested` classes to group related tests (e.g., by role or test type)
2. Use descriptive `@DisplayName` annotations explaining the test purpose
3. Use real JWT tokens via `@Value` injection (never `@WithMockUser`)
4. Include JavaDoc comments explaining business context and security implications
5. Follow AAA pattern (Arrange-Act-Assert) with clear section comments
6. Assert on both status codes and response structure when applicable"""
    
    def _build_guidelines_section(self, scenario: Dict[str, Any]) -> str:
        """Build the coding guidelines section."""
        data_level = scenario.get("data_awareness_level", "LOW")
        
        guidelines = """## CODING GUIDELINES AND CONSTRAINTS

### Test Class Structure
```java
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class {ServiceName}{Resource}Test {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Value("${test.jwt.admin}")
    private String adminToken;
    
    @Value("${test.jwt.user}")
    private String userToken;
    
    // Additional role tokens as needed
    
    @Nested
    @DisplayName("Authorization Tests - {ScenarioType}")
    class AuthorizationTests {
        // Test methods here
    }
}
```

### Required Test Coverage

**For Entry Point Scenarios**:
1. **Allowed Roles**: For each allowed role:
   - Authenticate with valid JWT token
   - Send request with valid body (if applicable)
   - Assert 2xx status code
   - Validate response structure if possible

2. **Denied Roles**: For each denied role:
   - Authenticate with valid JWT token
   - Send request
   - Assert 403 Forbidden status

3. **Unauthenticated**: 
   - Send request without any Authorization header
   - Assert 401 or 403 depending on security configuration

4. **Unsupported Roles**:
   - Test with a role not in the allowed list (e.g., GUEST)
   - Assert 403 Forbidden

**For Downstream Scenarios**:
1. **Consistent Path**:
   - Test that allowed roles can traverse full chain
   - Use realistic data enabling downstream execution
   - Assert success response from entry point

2. **Inconsistent Path**:
   - Create baseline test showing entry point behavior
   - Create tests demonstrating where policy diverges
   - Document inconsistency with detailed comments

### Code Quality Requirements

1. **Naming Convention**: Use descriptive method names like `methodName_asRole_returnsStatus`
2. **Documentation**: Every test must have a JavaDoc explaining:
   - What is being tested
   - Why the expected behavior is correct/incorrect
   - Any security implications
3. **Assertions**: Use specific assertions, not just status checks where possible
4. **Test Data**: Use realistic values, not placeholders like "test" or "123"
5. **Organization**: Group related tests in `@Nested` classes by concern

### Security Best Practices

- Never use `@WithMockUser` - always use real JWT tokens
- Include security-related comments documenting vulnerabilities
- Test both positive (access granted) and negative (access denied) cases
- For sensitive data endpoints, add tests validating data exposure boundaries"""
        
        if data_level == "HIGH":
            guidelines += """

### High Data Awareness Requirements (Downstream Scenarios)

For scenarios with depth > 0, you must ensure **test data coherence**:

1. **Request Body Construction**:
   - Use the Entity Schema to build valid request bodies
   - Include all required fields with realistic values
   - Ensure foreign key references are consistent (e.g., valid IDs)

2. **Path Variable Resolution**:
   - Use realistic identifiers that would exist in the system
   - Consider URL encoding for special characters
   - Document assumed entity state

3. **Query Parameters**:
   - Include all required query parameters
   - Use realistic filter/sort values
   - Test both minimal and complete parameter sets

4. **State Dependencies**:
   - If the operation depends on existing state (e.g., booking requires available seat),
     document the assumed preconditions
   - Consider ordering constraints in your test design
"""
        
        return guidelines
    
    def _build_output_specification(self, scenario: Dict[str, Any]) -> str:
        """Build the output specification section."""
        template_id = scenario.get("prompt_template_id", "")
        
        cot_instruction = ""
        if template_id == "downstream_inconsistency_cot":
            cot_instruction = """
### Chain-of-Thought Reasoning Process

Before writing the tests, explicitly reason through:

1. **Identify the Call Chain**: List all services in the call sequence
2. **Analyze Permissions at Each Node**: For each role, determine what is expected at each service
3. **Detect Divergence Points**: Identify where expected vs actual permissions differ
4. **Determine Test Strategy**: 
   - How to demonstrate the baseline behavior?
   - How to reveal the inconsistency?
   - What data is needed to trigger the full chain?
5. **Security Impact Assessment**: What are the implications of this inconsistency?

Include your reasoning in test comments to document the analysis."""
        
        return f"""## OUTPUT SPECIFICATION

### Generated Test Requirements

You must generate a **complete, compilable JUnit 5 test class** that:

1. **Compiles Without Errors**:
   - All imports included
   - All variables properly typed
   - All methods correctly structured

2. **Follows Spring Boot Testing Best Practices**:
   - Uses `@SpringBootTest` with `@AutoConfigureMockMvc`
   - Loads `test` profile via `@ActiveProfiles("test")`
   - Injects JWT tokens via `@Value` from environment properties

3. **Includes Comprehensive Test Coverage**:
   - All roles from the permission matrix are tested
   - Both allowed and denied scenarios covered
   - Edge cases (unauthenticated, unsupported roles) included

4. **Documents Security Context**:
   - JavaDoc comments explain authorization expectations
   - Inline comments document any detected vulnerabilities
   - References to security policies or compliance requirements

{cot_instruction}

### Response Format

Provide your response as a **single Java file content** containing the complete test class.

Structure:
```java
package com.example.test;

// All necessary imports

/**
 * Test class documentation
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class NameTest {{
    // Fields and setup
    
    @Nested
    class TestGroup {{
        // Test methods
    }}
}}
```

Do not include markdown code block markers in your response - provide clean Java code ready to be saved to a file."""
    
    def _format_entity_schema(self, s: Dict[str, Any]) -> str:
        """Format entity schema for the prompt."""
        schema = s.get("entity_schema")
        if not schema:
            return ""
        
        fields = schema.get("fields", [])
        if not fields:
            return ""
        
        lines = ["**Entity Schema** (use for constructing valid request bodies):", "```java"]
        lines.append(f"public class {schema.get('entity_name', 'Entity')} {{")
        
        for field in fields:
            fname = field.get("name")
            ftype = self._java_type(field.get("type"))
            
            # Add annotations
            for ann in field.get("annotations", []):
                aname = ann.get("name", "")
                attrs = ann.get("attributes", {})
                if attrs:
                    attr_str = ", ".join(
                        f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}"
                        for k, v in attrs.items()
                    )
                    lines.append(f"    @{aname}({attr_str})")
                else:
                    lines.append(f"    @{aname}")
            
            if fname:
                lines.append(f"    private {ftype} {fname};")
        
        lines.append("}")
        lines.append("```")
        
        return "\n".join(lines)
    
    def _java_type(self, py_type: Optional[str]) -> str:
        """Convert Python type to Java type."""
        if not py_type:
            return "String"
        
        t = py_type.lower()
        type_mapping = {
            "int": "int",
            "integer": "Integer",
            "long": "long",
            "double": "double",
            "float": "float",
            "boolean": "boolean",
            "bool": "boolean",
            "string": "String",
            "str": "String",
        }
        
        return type_mapping.get(t, "String")


class LangChainPromptChain:
    """LangChain-based prompt chain for long distributed paths.
    
    Uses LangChain's SequentialChain to implement a Map-Reduce pattern where
    each service in the call chain is analyzed separately (Map phase) and then
    results are combined into a coherent test strategy (Reduce phase).
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        self.builder = EnhancedPromptBuilder()
        self.llm = llm
        self._chains: Dict[str, Any] = {}
    
    def _create_phase_chain(
        self, 
        prompt_template: str, 
        output_key: str
    ) -> LLMChain:
        """Create an LLMChain for a specific phase."""
        prompt = PromptTemplate(
            input_variables=["scenario", "phase_context"],
            template=prompt_template,
        )
        
        # If no LLM provided, create a mock chain that returns the prompt
        if self.llm is None:
            return MockLLMChain(prompt=prompt, output_key=output_key)
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_key=output_key,
        )
    
    def _build_map_reduce_chain(self, scenario: Dict[str, Any]) -> SequentialChain:
        """Build a SequentialChain for Map-Reduce processing."""
        chain_perms = scenario.get("chain_permissions", {})
        
        if not chain_perms:
            # Simple chain for single endpoint
            return self._build_simple_chain(scenario)
        
        chains = []
        output_keys = []
        
        # Phase 1: Entry point analysis chain
        entry_template = self._create_entrypoint_template()
        entry_chain = self._create_phase_chain(
            "entrypoint", 
            entry_template, 
            "entrypoint_analysis"
        )
        chains.append(entry_chain)
        output_keys.append("entrypoint_analysis")
        
        # Phase 2+: Downstream service analysis chains (Map)
        service_list = list(chain_perms.items())
        for idx, (service_id, perms) in enumerate(service_list[1:], start=2):
            service_template = self._create_service_template(service_id, perms, idx, len(service_list))
            service_chain = self._create_phase_chain(
                f"service_{service_id}",
                service_template,
                f"service_{idx}_analysis"
            )
            chains.append(service_chain)
            output_keys.append(f"service_{idx}_analysis")
        
        # Final Phase: Synthesis chain (Reduce)
        synthesis_template = self._create_synthesis_template(scenario)
        synthesis_chain = self._create_phase_chain(
            "synthesis",
            synthesis_template,
            "final_test_class"
        )
        chains.append(synthesis_chain)
        output_keys.append("final_test_class")
        
        # Create sequential chain
        return SequentialChain(
            chains=chains,
            input_variables=["scenario"],
            output_variables=output_keys,
            verbose=True,
        )
    
    def _build_simple_chain(self, scenario: Dict[str, Any]) -> SequentialChain:
        """Build a simple chain for single endpoint scenarios."""
        template = self.builder.build(scenario)
        
        prompt = PromptTemplate(
            input_variables=["scenario"],
            template=template + "\n\nScenario: {scenario}",
        )
        
        if self.llm is None:
            chain = MockLLMChain(prompt=prompt, output_key="test_output")
        else:
            chain = LLMChain(llm=self.llm, prompt=prompt, output_key="test_output")
        
        return SequentialChain(
            chains=[chain],
            input_variables=["scenario"],
            output_variables=["test_output"],
        )
    
    def _create_entrypoint_template(self) -> str:
        """Create prompt template for entry point analysis."""
        return """# PHASE 1: ENTRY POINT ANALYSIS

Analyze the authorization behavior at the entry point of this distributed call chain.

**Scenario Data**: {scenario}

**Context**: {phase_context}

**Your Task**:
1. Design test methods that validate the entry point authorization matrix
2. Determine what request data is needed to trigger the downstream chain
3. Document the baseline expected behavior for each role
4. Identify any preconditions for successful downstream traversal

Output a structured analysis of the entry point authorization requirements."""
    
    def _create_service_template(
        self, 
        service_id: str, 
        permissions: Dict,
        phase: int,
        total_phases: int
    ) -> str:
        """Create prompt template for service analysis."""
        return f"""# PHASE {phase}/{total_phases}: DOWNSTREAM SERVICE ANALYSIS

Analyze authorization behavior at downstream service: **{service_id}**

**Service Data**: Available in scenario

**Context from Previous Phase**: {{phase_context}}

**Your Task**:
1. Compare permissions with the entry point baseline
2. Identify any inconsistencies
3. Design test assertions specific to this service
4. Document integration with the overall call chain

Output a structured analysis of this service's authorization behavior."""
    
    def _create_synthesis_template(self, scenario: Dict[str, Any]) -> str:
        """Create prompt template for final synthesis."""
        return """# FINAL PHASE: SYNTHESIS AND TEST GENERATION

Synthesize all phase analyses into a complete JUnit 5 test class.

**Scenario**: {scenario}

**Previous Analyses**: {phase_context}

**Instructions**:
1. Combine all phase analyses into a unified test class
2. Create @Nested classes for each concern
3. Include comprehensive JavaDoc
4. Use real JWT tokens via @Value
5. Test all roles and inconsistencies

Generate complete, compilable Java code."""
    
    def create_distributed_path_chain(self, scenario: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create a chain of prompts for analyzing distributed paths.
        
        Returns a list of sub-prompts, one for each service in the chain,
        plus a final synthesis prompt.
        """
        chain_perms = scenario.get("chain_permissions", {})
        inconsistencies = scenario.get("policy_inconsistencies", [])
        
        if not chain_perms:
            # Fall back to standard prompt if no chain
            return [{"type": "full", "prompt": self.builder.build(scenario)}]
        
        # Create sub-prompts for each service in the chain
        sub_prompts = []
        
        # Phase 1: Entry point analysis
        entry_point = self._create_entrypoint_analysis_prompt(scenario)
        sub_prompts.append({
            "type": "entrypoint",
            "phase": 1,
            "prompt": entry_point,
            "target": "entrypoint"
        })
        
        # Phase 2: Downstream service analysis (Map phase)
        service_list = list(chain_perms.items())
        for idx, (service_id, perms) in enumerate(service_list[1:], start=2):
            service_prompt = self._create_service_analysis_prompt(
                scenario, service_id, perms, idx, len(service_list)
            )
            sub_prompts.append({
                "type": "downstream",
                "phase": idx,
                "prompt": service_prompt,
                "target": service_id,
                "permissions": perms
            })
        
        # Phase 3: Synthesis/Reduce phase
        synthesis_prompt = self._create_synthesis_prompt(scenario, sub_prompts, inconsistencies)
        sub_prompts.append({
            "type": "synthesis",
            "phase": "final",
            "prompt": synthesis_prompt,
            "target": "all_services"
        })
        
        return sub_prompts
    
    def _create_entrypoint_analysis_prompt(self, scenario: Dict[str, Any]) -> str:
        """Create a focused prompt for entry point analysis."""
        endpoint = scenario.get("endpoint", "")
        method = scenario.get("method", "")
        allowed_roles = scenario.get("allowed_roles", [])
        denied_roles = scenario.get("denied_roles", [])
        
        return f"""# PHASE 1: ENTRY POINT ANALYSIS

Analyze the authorization behavior at the entry point of this distributed call chain.

**Endpoint**: {method} {endpoint}

**Roles to Test**:
- Allowed: {', '.join(allowed_roles) if allowed_roles else 'None'}
- Denied: {', '.join(denied_roles) if denied_roles else 'None'}

**Your Task**:
1. Design test methods that validate the entry point authorization matrix
2. Determine what request data is needed to trigger the downstream chain
3. Document the baseline expected behavior for each role
4. Identify any preconditions that must be met for successful downstream traversal

**Output**: A test design specification for the entry point that includes:
- Required JWT tokens per role
- Request body construction guidelines
- Expected status codes per role
- Data requirements for downstream execution"""
    
    def _create_service_analysis_prompt(
        self, 
        scenario: Dict[str, Any], 
        service_id: str, 
        permissions: Dict,
        phase: int,
        total_phases: int
    ) -> str:
        """Create a focused prompt for analyzing a specific downstream service."""
        allowed = permissions.get("allowed_roles", [])
        denied = permissions.get("denied_roles", [])
        unknown = permissions.get("unknown_roles", [])
        
        return f"""# PHASE {phase}/{total_phases}: DOWNSTREAM SERVICE ANALYSIS

Analyze authorization behavior at downstream service: **{service_id}**

**Service Permissions**:
- Allowed Roles: {', '.join(allowed) if allowed else 'None'}
- Denied Roles: {', '.join(denied) if denied else 'None'}
- Unknown/Undefined Roles: {', '.join(unknown) if unknown else 'None'}

**Your Task**:
1. Compare these permissions with the entry point baseline
2. Identify any inconsistencies (over-permissive, under-permissive, or undefined)
3. Design test assertions specific to this service
4. Document how this service fits in the overall call chain

**Context from Entry Point**:
- Entry roles: {', '.join(scenario.get("allowed_roles", []))}
- Expected to trigger: {scenario.get("total_calls", 0)} downstream calls

**Output**: Authorization analysis for {service_id} including:
- Inconsistency detection (if any)
- Role permission comparison with entry point
- Test assertions for this service
- Integration notes for the full chain"""
    
    def _create_synthesis_prompt(
        self, 
        scenario: Dict[str, Any], 
        sub_prompts: List[Dict],
        inconsistencies: List[Dict]
    ) -> str:
        """Create a synthesis prompt that combines all phase analyses."""
        inconsistency_summary = "\n".join(
            f"- {inc.get('type', 'Unknown')}: affects role {inc.get('role', 'N/A')}"
            for inc in inconsistencies
        ) if inconsistencies else "- No inconsistencies detected"
        
        return f"""# PHASE FINAL: SYNTHESIS AND TEST GENERATION

Synthesize the analyses from all phases into a complete, coherent test class.

**Scenario Overview**:
- Entry Point: {scenario.get("method", "")} {scenario.get("endpoint", "")}
- Total Services in Chain: {scenario.get("total_calls", 0)}
- Max Depth: {scenario.get("max_depth", 0)}

**Detected Inconsistencies**:
{inconsistency_summary}

**Chain Structure**:
{chr(10).join([f"- Phase {sp.get('phase', '?')}: {sp.get('target', 'unknown')}" for sp in sub_prompts if sp.get('type') != 'synthesis'])}

**Synthesis Instructions**:
1. Combine all phase analyses into a single, unified test class
2. Ensure consistent naming and documentation across all test methods
3. Create a test flow that demonstrates the complete authorization behavior:
   - Entry point acceptance
   - Downstream traversal
   - Inconsistency detection (if applicable)
   - Error handling at each layer

4. For inconsistencies found, create explicit test methods that:
   - Document the expected (correct) behavior
   - Demonstrate the actual (incorrect) behavior
   - Explain the security implications
   - Provide clear evidence for remediation

5. Include comprehensive JavaDoc explaining:
   - The overall call chain architecture
   - Authorization policy expectations
   - Any known issues or inconsistencies
   - Data requirements for full chain execution

**Output Requirements**:
Generate a complete, compilable JUnit 5 test class that:
- Uses @SpringBootTest + @AutoConfigureMockMvc + @ActiveProfiles("test")
- Injects JWT tokens via @Value (never @WithMockUser)
- Includes nested classes organizing tests by concern
- Provides clear, actionable test results for each role and service
- Documents all inconsistencies with specific evidence

Use the following structure:
```java
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class Distributed{scenario.get('endpoint', '/Resource').rstrip('/').split('/')[-1].replace('{', '').replace('}', '').title()}FlowTest {{
    // Fields and setup
    
    @Nested
    @DisplayName("Call Chain: {scenario.get('endpoint', '')}")
    class CallChainTests {{
        // Entry point tests
        // Downstream tests
        // Inconsistency tests (if applicable)
    }}
}}
```"""
    
    def execute_chain(self, scenario: Dict[str, Any]) -> str:
        """Execute the LangChain Map-Reduce chain for distributed paths.
        
        Returns the generated test code or the comprehensive prompt if no LLM available.
        """
        try:
            # Build the sequential chain
            chain = self._build_map_reduce_chain(scenario)
            
            # Execute the chain
            result = chain.run({"scenario": json.dumps(scenario)})
            
            # If using mock chain, return the comprehensive prompt
            if isinstance(result, dict) and "final_test_class" in result:
                return result["final_test_class"]
            elif isinstance(result, str):
                return result
            else:
                return str(result)
                
        except Exception as e:
            # Fallback to standard builder on error
            print(f"LangChain execution failed: {e}. Falling back to standard builder.")
            return self.builder.build(scenario)
    
    def _build_comprehensive_prompt(
        self, 
        chain_prompts: List[Dict[str, str]], 
        scenario: Dict[str, Any]
    ) -> str:
        """Build a comprehensive prompt from the chain of analyses."""
        # Combine entry point, downstream analyses, and synthesis
        sections = []
        
        # Header
        sections.append(self.builder._build_role_section())
        sections.append(self.builder._build_task_section(scenario))
        
        # Add the chain-of-thought structure from LangChain phases
        sections.append("## DISTRIBUTED PATH ANALYSIS (LangChain Map-Reduce Pattern)")
        sections.append("""
This scenario uses a **Map-Reduce prompt chain** to analyze the distributed call path:
1. **Map Phase**: Each service in the chain is analyzed independently for authorization behavior
2. **Reduce Phase**: Results are synthesized into a coherent test strategy
3. **Output**: A unified test class that validates the entire chain with consistent authorization
""")
        
        # Add each phase as a structured section
        for prompt_info in chain_prompts:
            if prompt_info.get("type") == "entrypoint":
                sections.append(f"\n### Phase 1: Entry Point Analysis\n{prompt_info.get('prompt', '')}")
            elif prompt_info.get("type") == "downstream":
                sections.append(f"\n### Phase {prompt_info.get('phase', '?')}: Service {prompt_info.get('target', '')}\n{prompt_info.get('prompt', '')}")
            elif prompt_info.get("type") == "synthesis":
                sections.append(f"\n### Final Phase: Synthesis\n{prompt_info.get('prompt', '')}")
        
        # Add business and path context
        business_ctx = self.builder.business_analyzer.analyze(scenario)
        path_ctx = self.builder.path_analyzer.analyze(scenario)
        sections.append(self.builder._build_business_context_section(business_ctx))
        sections.append(self.builder._build_path_structure_section(path_ctx))
        
        # Add guidelines
        sections.append(self.builder._build_guidelines_section(scenario))
        
        return "\n\n".join(filter(None, sections))


class AdaptivePromptExecutor:
    """Adaptive prompt executor for handling long distributed paths.
    
    Uses LangChain for very long distributed paths (complexity > 0.8)
    to implement Map-Reduce prompt chaining across service boundaries.
    """
    
    def __init__(self):
        self.builder = EnhancedPromptBuilder()
        self.langchain_executor = LangChainPromptChain() if LANGCHAIN_AVAILABLE else None
    
    def execute(self, scenario: Dict[str, Any]) -> str:
        """Execute adaptive prompt generation based on scenario complexity."""
        complexity_score = self._calculate_complexity(scenario)
        
        # Determine execution strategy based on complexity
        if complexity_score > 0.8:
            # Use LangChain for very complex, long distributed paths
            if self.langchain_executor and scenario.get("type") == "DownstreamPolicyConsistency":
                return self.langchain_executor.execute_chain(scenario)
            return self._chunked_prompt_strategy(scenario)
        elif complexity_score > 0.5:
            return self._summarized_prompt_strategy(scenario)
        else:
            return self.builder.build(scenario)
    
    def _calculate_complexity(self, scenario: Dict[str, Any]) -> float:
        """Calculate complexity score for the scenario."""
        score = 0.0
        
        # Depth-based complexity
        max_depth = scenario.get("max_depth", 0)
        score += min(max_depth / 5.0, 0.3)
        
        # Chain size complexity
        total_calls = scenario.get("total_calls", 0)
        score += min(total_calls / 10.0, 0.2)
        
        # Inconsistency complexity
        inconsistencies = len(scenario.get("policy_inconsistencies", []))
        score += min(inconsistencies / 5.0, 0.2)
        
        # Role count complexity
        role_count = len(scenario.get("allowed_roles", [])) + len(scenario.get("denied_roles", []))
        score += min(role_count / 10.0, 0.15)
        
        # Entity schema complexity
        if scenario.get("entity_schema"):
            fields = scenario.get("entity_schema", {}).get("fields", [])
            score += min(len(fields) / 20.0, 0.15)
        
        return min(score, 1.0)
    
    def _chunked_prompt_strategy(self, scenario: Dict[str, Any]) -> str:
        """Generate chunked prompts for very complex scenarios."""
        # Build base prompt
        base_prompt = self.builder.build(scenario)
        
        # If prompt is too long, create a focused version
        if len(base_prompt) > 8000:
            return self._create_focused_prompt(scenario)
        
        return base_prompt
    
    def _summarized_prompt_strategy(self, scenario: Dict[str, Any]) -> str:
        """Generate summarized prompt for moderately complex scenarios."""
        sections = [
            self.builder._build_role_section(),
            self.builder._build_task_section(scenario),
            self._build_summarized_scenario_section(scenario),
            self.builder._build_guidelines_section(scenario),
        ]
        
        return "\n\n".join(filter(None, sections))
    
    def _create_focused_prompt(self, scenario: Dict[str, Any]) -> str:
        """Create a focused prompt for the most critical aspects."""
        template_id = scenario.get("prompt_template_id", "")
        
        # Prioritize based on template type
        if "inconsistency" in template_id:
            focus = "inconsistency_detection"
        elif "public_sensitive" in template_id:
            focus = "security_vulnerability"
        else:
            focus = "authorization_matrix"
        
        return f"""# ROLE AND EXPERTISE

You are an Expert Software Test Engineer specializing in security testing and distributed systems authorization.

## PRIMARY TASK (FOCUSED)

Generate JUnit 5 tests for this {focus.replace("_", " ")} scenario.

## CRITICAL SCENARIO DETAILS

**Scenario ID**: {scenario.get("scenario_id", "")}
**Type**: {scenario.get("type", "")}
**Endpoint**: {scenario.get("method", "")} {scenario.get("endpoint", "")}
**Template**: {template_id}

{self._build_focused_content(scenario, focus)}

## ESSENTIAL GUIDELINES

1. Use `@SpringBootTest` + `@AutoConfigureMockMvc` + `@ActiveProfiles("test")`
2. Inject JWT tokens via `@Value`, never use `@WithMockUser`
3. Test all roles: {', '.join(scenario.get("allowed_roles", []) + scenario.get("denied_roles", []))}
4. Follow the reference patterns in your training
5. Include comprehensive JavaDoc comments

Generate complete, compilable Java test class now."""
    
    def _build_focused_content(self, scenario: Dict[str, Any], focus: str) -> str:
        """Build content focused on specific aspect."""
        if focus == "inconsistency_detection":
            inconsistencies = scenario.get("policy_inconsistencies", [])
            chain = scenario.get("chain_permissions", {})
            
            inc_summary = "\n".join(
                f"- {inc.get('type', '')}: role={inc.get('role', '')}"
                for inc in inconsistencies[:3]
            ) if inconsistencies else "- None"
            
            return f"""**Inconsistencies to Document**:
{inc_summary}

**Call Chain**: {', '.join(chain.keys())}

**Focus**: Create tests that clearly demonstrate where authorization policies diverge between services."""
        
        elif focus == "security_vulnerability":
            sensitivity = scenario.get("sensitivity_type", "")
            return f"""**SECURITY VULNERABILITY**:
- Sensitivity: {sensitivity}
- Issue: Sensitive endpoint is PUBLIC (no authentication required)

**Focus**: Document the vulnerability with tests showing unauthenticated access."""
        
        else:  # authorization_matrix
            expected = scenario.get("expected_status_by_role", {})
            status_str = "\n".join(f"- {role}: {status}" for role, status in expected.items())
            
            return f"""**Authorization Matrix**:
{status_str}

**Focus**: Validate that each role receives the expected status code."""
    
    def _build_summarized_scenario_section(self, scenario: Dict[str, Any]) -> str:
        """Build a summarized scenario section."""
        return f"""## SCENARIO SUMMARY

**ID**: {scenario.get("scenario_id", "")}
**Endpoint**: {scenario.get("method", "")} {scenario.get("endpoint", "")}
**Type**: {scenario.get("type", "")}
**Allowed**: {', '.join(scenario.get("allowed_roles", []))}
**Denied**: {', '.join(scenario.get("denied_roles", []))}
**Expected Status**: {scenario.get("expected_status_by_role", {})}
**Inconsistencies**: {len(scenario.get("policy_inconsistencies", []))}
"""


class PromptGeneratorFacade:
    """Facade providing a unified interface for prompt generation."""
    
    def __init__(self, use_adaptive: bool = True):
        self.use_adaptive = use_adaptive
        
        if self.use_adaptive:
            self.executor = AdaptivePromptExecutor()
        else:
            self.executor = EnhancedPromptBuilder()
    
    def generate(self, scenario: Dict[str, Any]) -> str:
        """Generate prompt for the given scenario."""
        if self.use_adaptive:
            return self.executor.execute(scenario)
        else:
            return self.executor.build(scenario)


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    """Load scenarios from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Endpoint filter for testing specific endpoints
# Format: {"endpoint_path": "HTTP_METHOD"}
# Example: {"/api/v1/orderservice/order": "PUT"}
ENDPOINT_FILTER: Dict[str, str] = {
    # Add endpoints here to filter, or leave empty to process all
    "/api/v1/orderservice/order": "PUT",
}


def generate_prompts(
    scenarios: List[Dict[str, Any]], 
    template_filter: Optional[str] = None,
    endpoint_filter: Optional[Dict[str, str]] = None,
    use_adaptive: bool = True
) -> List[Dict[str, str]]:
    """Generate prompts for all scenarios."""
    generator = PromptGeneratorFacade(use_adaptive=use_adaptive)
    items: List[Dict[str, str]] = []
    
    for scenario in scenarios:
        tid = scenario.get("prompt_template_id") or scenario.get("prompt_context", {}).get("template_id")
        endpoint = scenario.get("endpoint", "")
        method = scenario.get("method", "")
        
        # Apply template filter
        if template_filter and tid != template_filter:
            continue
        
        # Apply endpoint filter if provided
        if endpoint_filter:
            # Normalize endpoint path for comparison
            normalized_endpoint = endpoint.rstrip("/")
            filter_endpoint = None
            filter_method = None
            
            for ep, meth in endpoint_filter.items():
                normalized_filter_ep = ep.rstrip("/")
                # Check for exact match or pattern match (handling {?} wildcards)
                if normalized_endpoint == normalized_filter_ep:
                    filter_endpoint = ep
                    filter_method = meth
                    break
                # Handle wildcard patterns like /api/v1/cancelservice/cancel/{?}/{?}
                if "{?}" in normalized_filter_ep:
                    pattern_parts = normalized_filter_ep.split("/")
                    endpoint_parts = normalized_endpoint.split("/")
                    if len(pattern_parts) == len(endpoint_parts):
                        match = True
                        for p, e in zip(pattern_parts, endpoint_parts):
                            if p != "{?}" and p != e:
                                match = False
                                break
                        if match:
                            filter_endpoint = ep
                            filter_method = meth
                            break
            
            if filter_endpoint is None:
                continue
            
            # Check method match
            if filter_method and method.upper() != filter_method.upper():
                continue
        
        prompt = generator.generate(scenario)
        
        items.append({
            "scenario_id": scenario.get("scenario_id", ""),
            "prompt_template_id": tid or "",
            "endpoint": endpoint,
            "method": method,
            "prompt": prompt,
        })
    
    return items


def main() -> None:
    """Main entry point for prompt generation."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced LLM prompts for Java authorization tests"
    )
    parser.add_argument(
        "--scenarios",
        dest="scenarios_path",
        default="train-ticket-aitest/master/scenarios_llm_ready.json",
        help="Path to scenarios_llm_ready.json",
    )
    parser.add_argument(
        "--template-id",
        dest="template_id",
        help="Filter by prompt_template_id",
    )
    parser.add_argument(
        "--endpoint",
        dest="endpoint_filter",
        help='Filter by endpoint path and method (format: "METHOD /path/to/endpoint")',
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Output file for generated prompts (JSON lines)",
    )
    parser.add_argument(
        "--no-adaptive",
        dest="use_adaptive",
        action="store_false",
        default=True,
        help="Disable adaptive prompt executor",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about generated prompts",
    )
    args = parser.parse_args()
    
    scenarios_path = Path(args.scenarios_path)
    scenarios = load_scenarios(scenarios_path)
    
    # Parse endpoint filter if provided
    endpoint_filter = ENDPOINT_FILTER
    if args.endpoint_filter:
        parts = args.endpoint_filter.split(None, 1)
        if len(parts) == 2:
            endpoint_filter = {parts[1]: parts[0]}
    
    if args.verbose:
        print(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")
        if args.template_id:
            print(f"Filtering by template_id: {args.template_id}")
        if endpoint_filter:
            print(f"Filtering by endpoint: {endpoint_filter}")
        print(f"Adaptive mode: {args.use_adaptive}")
    
    prompts = generate_prompts(
        scenarios, 
        template_filter=args.template_id,
        endpoint_filter=endpoint_filter,
        use_adaptive=args.use_adaptive
    )
    
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(prompts)} prompts to {out_path}")
    else:
        for item in prompts:
            print(f"\n{'='*80}")
            print(f"Prompt for {item['scenario_id']} (template: {item['prompt_template_id']})")
            print(f"{'='*80}\n")
            print(item["prompt"])
            print("\n")


if __name__ == "__main__":
    main()

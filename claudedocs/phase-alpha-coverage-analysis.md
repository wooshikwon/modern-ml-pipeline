# Phase Alpha Coverage Analysis & Priority Matrix

## Current Coverage Baseline (2025-09-17)

### Overall Metrics
- **Total Coverage**: 49.63% (3,431 covered / 6,911 total lines)
- **Test Pass Rate**: 93.6% (1,212 passed / 1,295 executable tests)
- **Test Execution Time**: 4 minutes 34 seconds
- **Critical Issue**: TreeBasedFeatureGenerator fixed ✅

### Coverage Distribution by Module

#### High Priority Modules (Phase Beta Targets)

**🔴 P0 - Critical Infrastructure (Low Coverage, High Impact)**

| Module | Current | Target | Lines Gap | Impact | Effort |
|--------|---------|--------|-----------|---------|---------|
| `src/settings/factory.py` | 14.29% | 70% | +136 | Critical | High |
| `src/factory/factory.py` | 29.52% | 70% | +177 | Critical | High |
| `src/components/preprocessor/modules/feature_generator.py` | 58.97% | 80% | +16 | High | Low |

**🟡 P1 - Core Components (Medium Coverage, High Impact)**

| Module | Current | Target | Lines Gap | Impact | Effort |
|--------|---------|--------|-----------|---------|---------|
| `src/components/evaluator/modules/classification_evaluator.py` | 20.97% | 80% | +37 | High | Medium |
| `src/components/adapter/modules/sql_adapter.py` | 63.43% | 75% | +20 | Medium | Low |
| `src/components/adapter/modules/storage_adapter.py` | 76.92% | 85% | +7 | Medium | Low |

**🟢 P2 - Support Modules (Variable Coverage)**

| Module | Current | Target | Lines Gap | Impact | Effort |
|--------|---------|--------|-----------|---------|---------|
| `src/cli/utils/config_builder.py` | 78.14% | 85% | +13 | Medium | Low |
| `src/cli/utils/recipe_builder.py` | 42.45% | 60% | +43 | Low | Medium |
| `src/cli/utils/interactive_ui.py` | 37.80% | 60% | +18 | Low | Medium |

#### Already Achieving Target (✅ Maintain)

| Module | Coverage | Status |
|--------|----------|---------|
| `src/settings/config.py` | 100% | ✅ Excellent |
| `src/settings/recipe.py` | 100% | ✅ Excellent |
| `src/components/preprocessor/preprocessor.py` | 87.80% | ✅ Good |
| `src/components/adapter/registry.py` | 89.29% | ✅ Good |

### Priority Ranking Algorithm

**Impact Score Calculation:**
- Critical: Infrastructure, Factory patterns, Core pipelines (×3)
- High: Data adapters, Evaluators, Preprocessors (×2)
- Medium: CLI utilities, Template engines (×1.5)
- Low: Documentation, Utilities, Optional features (×1)

**Effort Score Calculation:**
- Low: <20 lines gap, simple logic, existing test patterns (×1)
- Medium: 20-50 lines gap, moderate complexity (×2)
- High: >50 lines gap, complex logic, new test patterns (×3)

**Priority Score = Impact Score / Effort Score**

### Phase Alpha to Beta Roadmap

#### Immediate Phase Beta Targets (Week 1-2)

**Target 1: Factory Pattern Foundation**
- `src/settings/factory.py`: 14.29% → 70% (+136 lines)
- Strategy: Unit tests with Real Object Testing philosophy
- Context: Use `settings_builder` fixture, no internal mocking
- Estimated effort: 8-12 hours

**Target 2: Core Factory Implementation**
- `src/factory/factory.py`: 29.52% → 70% (+177 lines)
- Strategy: Integration tests with ComponentTestContext
- Focus: Component creation workflows, error handling
- Estimated effort: 12-16 hours

#### Secondary Targets (Week 3)

**Target 3: Evaluator Enhancement**
- `src/components/evaluator/modules/classification_evaluator.py`: 20.97% → 80%
- Strategy: Integration tests with real model predictions
- Focus: Metric calculation accuracy, edge cases
- Estimated effort: 4-6 hours

**Target 4: Adapter Completion**
- `src/components/adapter/modules/sql_adapter.py`: 63.43% → 75%
- Strategy: Integration tests with DatabaseTestContext
- Focus: SQL query validation, error handling
- Estimated effort: 3-4 hours

### Coverage Gap Analysis

#### Lines Requiring Coverage (Top 10)

| Module | Missing Lines | Primary Gaps |
|--------|---------------|--------------|
| `factory.py` | 308 | Component creation, caching, error paths |
| `factory.py` (settings) | 210 | MLflow integration, validation |
| `recipe_builder.py` | 141 | Interactive recipe generation |
| `sql_adapter.py` | 64 | Query execution, connection handling |
| `interactive_ui.py` | 51 | User interaction flows |
| `classification_evaluator.py` | 49 | Metric computation, edge cases |
| `template_engine.py` | 19 | Jinja2 template processing |
| `storage_adapter.py` | 21 | File format handling |

### Testing Strategy Alignment

#### Real Object Testing Compliance
- ✅ No internal component mocking
- ✅ Use Context classes (MLflow, Component, Database)
- ✅ Test public API contracts only
- ✅ Deterministic execution with UUID naming

#### Performance Targets
- ✅ Unit tests: <100ms per test
- ✅ Integration tests: <1s per test
- ⚠️ Full suite: 4m34s (target: <15min) - **GOOD**

### Risk Assessment

#### High Risk Areas
1. **Factory Pattern Complexity**: Multiple creation strategies, caching
2. **MLflow Integration**: External dependency, state management
3. **SQL Adapter Security**: Query validation, injection prevention
4. **Interactive UI**: Complex user flows, hard to test

#### Mitigation Strategies
1. **Component Testing**: Break factory into smaller testable units
2. **MLflow Mocking**: Mock external MLflow calls only
3. **SQL Validation**: Test query parsing separately from execution
4. **UI Automation**: Use step-by-step scenario testing

### Success Metrics

#### Phase Alpha Exit Criteria ✅
- [x] Zero critical test failures (TreeBasedFeatureGenerator fixed)
- [x] Accurate coverage baseline established (49.63%)
- [x] Priority matrix completed
- [x] Infrastructure optimized (4m34s execution time acceptable)

#### Phase Beta Entry Criteria
- [ ] Factory modules reach 70% coverage
- [ ] Core evaluator modules reach 80% coverage
- [ ] Test suite maintains <15 minute execution
- [ ] 95% tests/README.md compliance

### Conclusion

Current state is strong with 49.63% coverage and 93.6% pass rate. Main coverage gaps are in:

1. **Factory patterns** (highest impact)
2. **Component evaluators** (core business logic)
3. **CLI utilities** (user experience)

The identified priority matrix provides a clear path to achieve the 85% coverage target while maintaining test quality standards defined in tests/README.md.

---

*Analysis Date: 2025-09-17*
*Coverage Tool: pytest-cov*
*Test Framework: pytest with custom Context classes*
*Testing Philosophy: Real Object Testing (tests/README.md)*
# Contexts

- Minimal contract & anti-patterns are in the refactoring analysis doc.
- Each test must create fresh Settings/Factory.
- Use file://{temp_dir}/mlruns and uuid-based experiment names.



## Performance Benchmark Example

Use the provided performance_benchmark fixture:

```python
def test_mlflow_context_init_time(mlflow_test_context, performance_benchmark):
    with performance_benchmark.measure_time("mlflow_context_init"):
        with mlflow_test_context.for_classification(experiment="bench") as ctx:
            assert ctx.experiment_exists()
    performance_benchmark.assert_time_under("mlflow_context_init", 0.12)
```

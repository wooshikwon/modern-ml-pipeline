#!/usr/bin/env python
"""Test script for get-config command with various env names."""

import subprocess
import sys
from pathlib import Path

def test_get_config(env_name):
    """Test get-config with a specific environment name."""
    print(f"\n=== Testing get-config for env_name: {env_name} ===")
    
    # Clean up any existing files
    config_file = Path(f"configs/{env_name}.yaml")
    env_file = Path(f".env.{env_name}.template")
    
    if config_file.exists():
        config_file.unlink()
        print(f"Removed existing {config_file}")
    
    if env_file.exists():
        env_file.unlink()
        print(f"Removed existing {env_file}")
    
    # Create input for interactive prompts
    # This simulates user selections for a minimal configuration
    inputs = "\n".join([
        # Data source: PostgreSQL (option 1)
        "1",
        # DB host
        "localhost",
        # DB port
        "5432",
        # DB name
        "test_db",
        # DB user
        "postgres",
        # MLflow enabled?
        "n",
        # Feature store enabled?
        "n",
        # Storage type: Local (option 1)
        "1",
        # Storage path
        "./data",
        # API Serving enabled?
        "y",
        # Workers
        "2",
        # Model stage: None (option 1)
        "1",
        # Auth enabled?
        "n",
        # Hyperparameter tuning enabled?
        "y",
        # Tuning engine: Optuna (option 1)
        "1",
        # Timeout
        "600",
        # Jobs
        "4",
        # Direction: Maximize (option 1)
        "1",
        # Monitoring enabled?
        "y",
        # Prometheus port
        "9091",
        # Grafana enabled?
        "y",
        # Grafana port
        "3001",
        # Advanced settings?
        "n",
    ])
    
    # Run the command with input
    result = subprocess.run(
        ["uv", "run", "mmp", "get-config", "--env-name", env_name],
        input=inputs,
        text=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Check if files were created
    if config_file.exists():
        print(f"✅ Created {config_file}")
        # Show first few lines
        with open(config_file) as f:
            lines = f.readlines()[:10]
            print("First 10 lines of config:")
            for line in lines:
                print(f"  {line.rstrip()}")
    else:
        print(f"❌ {config_file} was not created")
        return False
    
    if env_file.exists():
        print(f"✅ Created {env_file}")
        # Show first few lines
        with open(env_file) as f:
            lines = f.readlines()[:10]
            print("First 10 lines of .env template:")
            for line in lines:
                print(f"  {line.rstrip()}")
    else:
        print(f"❌ {env_file} was not created")
        return False
    
    return True

def main():
    """Test with various environment names."""
    test_cases = [
        "wooshik-test",
        "experiment-v3",
        "qa-environment",
        "feature-branch-123",
    ]
    
    all_passed = True
    for env_name in test_cases:
        if not test_get_config(env_name):
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
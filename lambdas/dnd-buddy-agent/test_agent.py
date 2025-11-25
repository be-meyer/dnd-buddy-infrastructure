#!/usr/bin/env python3
"""
Test script for D&D Buddy Agent running locally.
"""
import requests
import json
import sys

# Agent endpoint
AGENT_URL = "http://localhost:8080"

def test_ping():
    """Test health check endpoint."""
    print("Testing /ping endpoint...")
    response = requests.get(f"{AGENT_URL}/ping")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_invocation(user_id: str, campaign: str, prompt: str):
    """Test agent invocation."""
    print(f"Testing /invocations endpoint...")
    print(f"User: {user_id}, Campaign: {campaign}")
    print(f"Prompt: {prompt}\n")
    
    payload = {
        "input": {
            "userId": user_id,
            "campaign": campaign,
            "prompt": prompt
        }
    }
    
    response = requests.post(
        f"{AGENT_URL}/invocations",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}\n")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False

def main():
    """Run tests."""
    print("=" * 60)
    print("D&D Buddy Agent - Local Test")
    print("=" * 60 + "\n")
    
    # Test health check
    if not test_ping():
        print("❌ Health check failed. Is the agent running?")
        sys.exit(1)
    
    print("✅ Health check passed\n")
    print("-" * 60 + "\n")
    
    # Test invocations
    test_cases = [
        {
            "user_id": "user123",
            "campaign": "lost-mines",
            "prompt": "Tell me about the NPCs in Phandalin"
        },
        {
            "user_id": "user123",
            "campaign": "lost-mines",
            "prompt": "What monsters have we encountered?"
        },
        {
            "user_id": "user123",
            "campaign": "lost-mines",
            "prompt": "Summarize our last session"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        success = test_invocation(**test_case)
        
        if success:
            print(f"✅ Test case {i} passed\n")
        else:
            print(f"❌ Test case {i} failed\n")
        
        print("-" * 60 + "\n")
    
    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to agent. Make sure it's running on http://localhost:8080")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)

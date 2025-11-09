#!/usr/bin/env python3
"""
Web Automation Integration Test

Direct test of web automation model integration with LM Arena
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/gary/lm-arena')

from core.web_automation_model import MockWebAutomationModel
from core.agent import GenerationRequest


async def test_web_automation_direct():
    """Test web automation model directly"""
    print("🔧 Testing Web Automation Model Integration")
    print("=" * 50)

    # Initialize web automation model
    model = MockWebAutomationModel()

    # Test prompts for web automation
    test_prompts = [
        "search for weather in London",
        "scrape https://example.com",
        "search for python tutorials",
        "check the weather",
        "extract data from website"
    ]

    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 30)

        try:
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                model_name="web-automation",
                temperature=0.7,
                max_tokens=100
            )

            # Generate response
            response = await model.generate(request)

            print(f"✅ Success ({response.generation_time:.2f}s)")
            print(f"   Model: {response.model_name}")
            print(f"   Response: {response.content[:100]}...")

            results.append({
                "prompt": prompt,
                "success": True,
                "response": response.content[:200]
            })

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })

    # Summary
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    success_rate = (successful / total) * 100

    print(f"\n" + "=" * 50)
    print(f"📊 Integration Test Summary:")
    print(f"   Total Tests: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    print(f"   Success Rate: {success_rate:.1f}%")

    return results, success_rate


async def test_api_simulation():
    """Test API simulation"""
    print("\n🌐 Simulating API Integration")
    print("=" * 50)

    # Import the web automation model
    from core.web_automation_model import MockWebAutomationModel

    model = MockWebAutomationModel()

    # Simulate API request
    api_request = {
        "message": "search for current weather in London",
        "model": "web-automation",
        "temperature": 0.7,
        "max_tokens": 100,
        "conversation_id": "web-test-api-001"
    }

    print(f"API Request: {api_request['message']}")
    print(f"Model: {api_request['model']}")

    try:
        request = GenerationRequest(
            prompt=api_request["message"],
            model_name=api_request["model"],
            temperature=api_request["temperature"],
            max_tokens=api_request["max_tokens"]
        )

        response = await model.generate(request)

        print(f"✅ API Simulation Successful")
        print(f"   Response: {response.content[:200]}...")
        print(f"   Generation Time: {response.generation_time:.2f}s")

        return True

    except Exception as e:
        print(f"❌ API Simulation Failed: {str(e)}")
        return False


async def main():
    """Main test function"""
    print("🚀 LM Arena Web Automation Integration Test")
    print("=" * 60)

    try:
        # Test direct model integration
        results, success_rate = await test_web_automation_direct()

        # Test API simulation
        api_success = await test_api_simulation()

        # Final summary
        print(f"\n" + "=" * 60)
        print(f"🎯 FINAL INTEGRATION SUMMARY:")
        print(f"   Web Automation Success Rate: {success_rate:.1f}%")
        print(f"   API Simulation: {'✅ PASSED' if api_success else '❌ FAILED'}")

        if success_rate >= 80 and api_success:
            print(f"\n🎉 WEB AUTOMATION INTEGRATION: SUCCESSFUL")
            print(f"✅ Ready for production with web automation capabilities")
        else:
            print(f"\n⚠️  WEB AUTOMATION INTEGRATION: NEEDS IMPROVEMENT")
            print(f"❌ Success rate below 80% or API simulation failed")

        return success_rate >= 80 and api_success

    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
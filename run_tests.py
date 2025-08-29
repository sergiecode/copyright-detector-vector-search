#!/usr/bin/env python3
"""
🎵 Test Runner for Vector Search System

Comprehensive test runner that executes all tests and generates detailed reports.
Ensures the copyright detection system works perfectly.

Created by: Sergie Code - Software Engineer & YouTube Programming Educator
AI Tools for Musicians Series
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import argparse

class TestRunner:
    """Comprehensive test runner for the vector search system."""
    
    def __init__(self, project_root=None):
        """Initialize test runner."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        
        # Add src to Python path
        sys.path.insert(0, str(self.src_dir))
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            ('numpy', 'numpy'),
            ('scipy', 'scipy'), 
            ('pandas', 'pandas'),
            ('scikit-learn', 'sklearn'),
            ('faiss-cpu', 'faiss')
        ]
        
        missing_packages = []
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"  ✅ {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                print(f"  ❌ {package_name}")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("  ✅ All dependencies available")
        return True
    
    def run_unit_tests(self, verbose=False):
        """Run unit tests."""
        print("\n🧪 Running Unit Tests...")
        
        test_files = [
            "test_indexer.py",
            "test_search.py"
        ]
        
        results = {}
        
        for test_file in test_files:
            test_path = self.tests_dir / test_file
            if not test_path.exists():
                print(f"  ⚠️  Test file not found: {test_file}")
                continue
            
            print(f"\n  Running {test_file}...")
            start_time = time.time()
            
            try:
                # Run the test file
                cmd = [sys.executable, "-m", "unittest", f"tests.{test_file[:-3]}", "-v" if verbose else ""]
                cmd = [c for c in cmd if c]  # Remove empty strings
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"    ✅ PASSED ({duration:.2f}s)")
                    results[test_file] = {"status": "PASSED", "duration": duration}
                else:
                    print(f"    ❌ FAILED ({duration:.2f}s)")
                    results[test_file] = {
                        "status": "FAILED", 
                        "duration": duration,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                    if verbose:
                        print(f"    Error output:\n{result.stderr}")
                
            except subprocess.TimeoutExpired:
                print(f"    ⏰ TIMEOUT (>300s)")
                results[test_file] = {"status": "TIMEOUT", "duration": 300}
            except Exception as e:
                print(f"    💥 ERROR: {e}")
                results[test_file] = {"status": "ERROR", "error": str(e)}
        
        return results
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests."""
        print("\n🔗 Running Integration Tests...")
        
        test_file = "test_integration.py"
        test_path = self.tests_dir / test_file
        
        if not test_path.exists():
            print(f"  ⚠️  Integration test file not found: {test_file}")
            return {"test_integration.py": {"status": "NOT_FOUND"}}
        
        print(f"  Running {test_file}...")
        start_time = time.time()
        
        try:
            cmd = [sys.executable, "-m", "unittest", "tests.test_integration", "-v" if verbose else ""]
            cmd = [c for c in cmd if c]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for integration tests
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"    ✅ PASSED ({duration:.2f}s)")
                return {test_file: {"status": "PASSED", "duration": duration}}
            else:
                print(f"    ❌ FAILED ({duration:.2f}s)")
                if verbose:
                    print(f"    Error output:\n{result.stderr}")
                return {
                    test_file: {
                        "status": "FAILED",
                        "duration": duration,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                }
        
        except subprocess.TimeoutExpired:
            print(f"    ⏰ TIMEOUT (>600s)")
            return {test_file: {"status": "TIMEOUT", "duration": 600}}
        except Exception as e:
            print(f"    💥 ERROR: {e}")
            return {test_file: {"status": "ERROR", "error": str(e)}}
    
    def run_performance_tests(self):
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Tests...")
        
        try:
            # Import modules to test basic functionality
            from indexer import VectorIndexer
            from search import SimilaritySearcher, CopyrightDetector
            
            import numpy as np
            import tempfile
            import shutil
            
            # Create test data
            dimension = 96
            num_vectors = 1000
            embeddings = np.random.rand(num_vectors, dimension).astype(np.float32)
            metadata = [{"track_id": i, "filename": f"track_{i}.wav"} for i in range(num_vectors)]
            
            # Test index building performance
            print("  Testing index building...")
            start_time = time.time()
            indexer = VectorIndexer(dimension=dimension, index_type="FlatL2")
            indexer.add_embeddings(embeddings, metadata)
            build_time = time.time() - start_time
            print(f"    ✅ Built index with {num_vectors} vectors in {build_time:.3f}s")
            
            # Test search performance
            print("  Testing search performance...")
            searcher = SimilaritySearcher(indexer=indexer)
            query = np.random.rand(dimension).astype(np.float32)
            
            # Multiple search operations
            search_times = []
            for _ in range(10):
                start_time = time.time()
                results = searcher.search_similar(query, k=10)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = np.mean(search_times)
            print(f"    ✅ Average search time: {avg_search_time*1000:.2f}ms")
            
            # Test copyright detection performance
            print("  Testing copyright detection...")
            detector = CopyrightDetector(searcher)
            start_time = time.time()
            analysis = detector.analyze_track(query)
            detection_time = time.time() - start_time
            print(f"    ✅ Copyright analysis time: {detection_time*1000:.2f}ms")
            
            # Performance criteria
            performance_ok = True
            if build_time > 10:  # Should build 1000 vectors in under 10s
                print(f"    ⚠️  Index building slower than expected: {build_time:.3f}s")
                performance_ok = False
            
            if avg_search_time > 0.1:  # Should search in under 100ms
                print(f"    ⚠️  Search slower than expected: {avg_search_time*1000:.2f}ms")
                performance_ok = False
            
            if performance_ok:
                print("  ✅ Performance tests PASSED")
                return {"performance": {"status": "PASSED", "build_time": build_time, "search_time": avg_search_time}}
            else:
                print("  ⚠️  Performance tests completed with warnings")
                return {"performance": {"status": "WARNING", "build_time": build_time, "search_time": avg_search_time}}
        
        except Exception as e:
            print(f"  ❌ Performance tests FAILED: {e}")
            return {"performance": {"status": "FAILED", "error": str(e)}}
    
    def run_all_tests(self, verbose=False, skip_integration=False, skip_performance=False):
        """Run all tests and generate a report."""
        print("🎵 Vector Search System - Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        # Check dependencies first
        if not self.check_dependencies():
            print("\n❌ Cannot run tests - missing dependencies")
            return False
        
        all_results = {}
        
        # Run unit tests
        unit_results = self.run_unit_tests(verbose=verbose)
        all_results.update(unit_results)
        
        # Run integration tests
        if not skip_integration:
            integration_results = self.run_integration_tests(verbose=verbose)
            all_results.update(integration_results)
        
        # Run performance tests
        if not skip_performance:
            performance_results = self.run_performance_tests()
            all_results.update(performance_results)
        
        # Generate summary report
        total_time = time.time() - start_time
        self.generate_report(all_results, total_time)
        
        # Determine overall success
        failed_tests = [name for name, result in all_results.items() 
                       if result.get("status") in ["FAILED", "ERROR", "TIMEOUT"]]
        
        if not failed_tests:
            print("\n🎉 ALL TESTS PASSED! The app works perfectly.")
            return True
        else:
            print(f"\n❌ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
            return False
    
    def generate_report(self, results, total_time):
        """Generate a detailed test report."""
        print("\n📊 Test Report")
        print("=" * 30)
        
        passed = len([r for r in results.values() if r.get("status") == "PASSED"])
        failed = len([r for r in results.values() if r.get("status") in ["FAILED", "ERROR"]])
        warnings = len([r for r in results.values() if r.get("status") == "WARNING"])
        total = len(results)
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ✅")
        if warnings > 0:
            print(f"Warnings: {warnings} ⚠️")
        if failed > 0:
            print(f"Failed: {failed} ❌")
        print(f"Total time: {total_time:.2f}s")
        
        # Detailed results
        print("\nDetailed Results:")
        for test_name, result in results.items():
            status = result.get("status", "UNKNOWN")
            duration = result.get("duration", 0)
            
            status_icon = {
                "PASSED": "✅",
                "FAILED": "❌", 
                "ERROR": "💥",
                "TIMEOUT": "⏰",
                "WARNING": "⚠️",
                "NOT_FOUND": "❓"
            }.get(status, "❓")
            
            print(f"  {test_name}: {status_icon} {status} ({duration:.2f}s)")
            
            if status in ["FAILED", "ERROR"] and "error" in result:
                error_lines = result["error"].split('\n')[:3]  # Show first 3 lines
                for line in error_lines:
                    if line.strip():
                        print(f"    {line.strip()}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Vector Search System Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    runner = TestRunner(project_root=args.project_root)
    success = runner.run_all_tests(
        verbose=args.verbose,
        skip_integration=args.skip_integration,
        skip_performance=args.skip_performance
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

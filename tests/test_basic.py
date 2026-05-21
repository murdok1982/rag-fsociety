"""Basic tests for rag-fsociety."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    try:
        import chromadb
        import sentence_transformers
        print(f"chromadb available")
        print(f"sentence-transformers available")
    except ImportError as e:
        assert False, f"Missing dependency: {e}"


def test_config_exists():
    for f in ["chat_rag.py", "index_security_data.py"]:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f)
        assert os.path.exists(path), f"{f} not found"
        print(f"{f} found")


if __name__ == "__main__":
    test_imports()
    test_config_exists()
    print("All basic tests passed!")

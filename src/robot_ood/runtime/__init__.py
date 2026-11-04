"""What executes inside a rollout: the belief filter, the retrieval arm, and the blend they decide.

Docstring-only on purpose: a re-export here would run the whole layer -- and torch with it -- for
every importer of any one module in it. See tests/test_imports.py.
"""
